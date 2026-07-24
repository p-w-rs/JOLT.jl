# =====================================================================
# Shape-shuffling primitives — reshape and transpose (permutedims).
#
# Their backward passes are just the inverse shuffle, so each vjp is the same
# op with an inverted parameter. Non-tensor params (target shape, permutation)
# live in the op struct.
# =====================================================================

# --- transpose / permutedims ----------------------------------------
# perm is a 1-based permutation of the input dims (StableHLO wants it 0-based).
struct TransposeOp <: Op
    perm::Vector{Int}
end
function outshape(op::TransposeOp, sa)
    (length(op.perm) == length(sa) && isperm(op.perm)) ||
        error("transpose: $(op.perm) is not a permutation of 1:$(length(sa))")
    return ntuple(i -> sa[op.perm[i]], length(op.perm))
end
lower(op::TransposeOp, a)     = shlo.transpose(a.value;   # remap Julia perm into reversed-dim space
    permutation = i64array([length(op.perm) - p for p in reverse(op.perm)]))
vjp(op::TransposeOp, ȳ, ins, out) = (apply(TransposeOp(invperm(op.perm)), ȳ),)

Base.permutedims(a::AbstractTensor, perm) = apply(TransposeOp(collect(Int, perm)), a)
Base.permutedims(a::AbstractTensor)       = permutedims(a, reverse(1:ndims(a)))
Base.adjoint(a::AbstractTensor)           = permutedims(a)  # real tensors: adjoint ≡ transpose

# --- reshape --------------------------------------------------------
struct ReshapeOp <: Op
    shape::Tuple           # target dims (Int | symbolic)
end
function outshape(op::ReshapeOp, sa)
    for d in op.shape                             # reject negative targets before they mint invalid MLIR
        d isa Int && d < 0 &&
            error(d == -1 ?
                "reshape: dimension inference (-1) is not supported — give explicit sizes" :
                "reshape: target dim $d is negative; concrete dims must be ≥ 0")
    end
    # Element counts must be PROVABLY equal — symbolic dims included (prod is a
    # Dim, dims_equal handles the Poly algebra). Strict, like every other op.
    dims_equal(prod(sa), prod(op.shape)) ||
        error("reshape: $(sa) ($(prod(sa)) elems) cannot become $(op.shape) ($(prod(op.shape)) elems)")
    return op.shape
end
lower(op::ReshapeOp, a) = shlo.reshape(a.value; result_0 = mlir_type(eltype(a), op.shape))
vjp(op::ReshapeOp, ȳ, ins, out) = (apply(ReshapeOp(size(ins[1])), ȳ),)   # reshape back

Base.reshape(a::AbstractTensor, dims::Union{Integer,Symbol,Poly}...) =
    apply(ReshapeOp(map(todim, dims)), a)

# =====================================================================
# Broadcasting — .+ / .- / .* / ./  (NumPy/TF/JAX semantics).
#
# `broadcast_to` (stablehlo.broadcast_in_dim) stretches an operand to a target
# shape: input dim i maps to target dim `bdims[i]`; target dims NOT in `bdims`
# are new axes the operand is replicated across. `.+`/`.-`/`.*`/`./` broadcast
# BOTH operands to the common shape (dims.jl's broadcast_shapes) then reuse the
# strict same-shape op — so their gradients come for free from broadcast_to's
# vjp (a reduce_sum over the stretched/new axes) composed with the base op.
# LEFT-pad alignment (NumPy/TF/JAX), NOT Julia's leading-dim alignment:
#     (32, 10) .+ (10,)  ->  (32, 10) .+ (1, 10)  ->  bias added to each row.
# Non-fused: each dotted call emits its own op (the backend fuses).
# =====================================================================
struct BroadcastToOp <: Op
    shape::Tuple           # target shape
    bdims::Vector{Int}     # 1-based: input dim i -> target dim bdims[i]
    # DYNAMIC broadcast plan (empty ⇒ fully static, lower via broadcast_in_dim):
    # one entry per target axis (Julia order) giving its runtime size — a static
    # Int, or (src::AbstractTensor, src_julia_axis) to read at run time via
    # get_dimension_size. Set by `broadcast_to` when a dynamic target axis isn't
    # supplied by the operand itself. See `broadcast_to`.
    dynsize::Vector{Any}
end
BroadcastToOp(shape, bdims) = BroadcastToOp(shape, bdims, Any[])
function outshape(op::BroadcastToOp, sa)
    n = length(op.shape)
    length(op.bdims) == length(sa) ||
        error("broadcast_to: $(length(op.bdims)) bdims for a rank-$(length(sa)) input")
    # bdims must be increasing, in-range, unique — this keeps the vjp's reshape-back
    # sound (a permuting broadcast would need a transpose, which we don't emit).
    (all(b -> 1 <= b <= n, op.bdims) && allunique(op.bdims) && issorted(op.bdims)) ||
        error("broadcast_to: bdims $(op.bdims) must be unique, in 1:$n, and increasing")
    for (i, d) in enumerate(sa)
        t = op.shape[op.bdims[i]]
        (dims_equal(d, 1) || dims_equal(d, t)) ||
            error("broadcast_to: input dim $i ($d) cannot stretch to target dim $(op.bdims[i]) ($t)")
    end
    # A dynamic target axis not carried by the operand is fine now — `broadcast_to`
    # has already resolved a runtime-size source for it (or errored). No rejection.
    return op.shape
end

# Static case: broadcast_in_dim (extents baked into the result type). Dynamic case:
# dynamic_broadcast_in_dim, with output_dimensions built as a 1-D i32 shape vector
# (in REVERSED/MLIR axis order) from get_dimension_size of the recorded sources +
# static-dim constants. (Verified against iree-compile 3.12; see [[jolt-verified-constraints]].)
function lower(op::BroadcastToOp, x)
    bdim_attr = i64array([length(op.shape) - b for b in reverse(op.bdims)])
    isempty(op.dynsize) && return shlo.broadcast_in_dim(x.value;
        result_0 = mlir_type(eltype(x), op.shape), broadcast_dimensions = bdim_attr)
    n = length(op.shape); blk = session().block; one_i32 = mlir_type(Int32, (1,))
    parts = IR.Value[]
    for k in 0:(n - 1)                          # MLIR result axis k ⇐ Julia target axis n-k
        spec = op.dynsize[n - k]
        if spec isa Int
            c = shlo.constant(; value = IR.DenseElementsAttribute(Int32[spec]), output = one_i32)
            push!(blk, c); push!(parts, IR.result(c, 1))
        else
            src, a = spec                       # Julia axis a of src ⇒ MLIR axis ndims(src)-a
            gd = shlo.get_dimension_size(src.value; dimension = ir_attr("$(ndims(src) - a) : i64"))
            push!(blk, gd)
            rs = shlo.reshape(IR.result(gd, 1); result_0 = one_i32)   # scalar i32 → tensor<1xi32>
            push!(blk, rs); push!(parts, IR.result(rs, 1))
        end
    end
    dims = shlo.concatenate(parts; result_0 = mlir_type(Int32, (n,)), dimension = ir_attr("0 : i64"))
    push!(blk, dims)
    return shlo.dynamic_broadcast_in_dim(x.value, IR.result(dims, 1);
        result_0 = mlir_type(eltype(x), op.shape), broadcast_dimensions = bdim_attr)
end
# backward: sum ȳ over the axes that were replicated (new) or stretched (input
# was size-1), then reshape back so the input's size-1 dims are restored. (Works
# unchanged for the dynamic case — reduce_sum handles dynamic dims, and its own
# vjp broadcasts back with a source, closing the loop for higher order.)
function vjp(op::BroadcastToOp, ȳ, ins, out)
    insh = size(ins[1])
    new_ax     = [j for j in 1:length(op.shape) if !(j in op.bdims)]
    stretched  = [op.bdims[i] for i in eachindex(insh)
                  if dims_equal(insh[i], 1) && !dims_equal(op.shape[op.bdims[i]], 1)]
    R = sort!(unique(vcat(new_ax, stretched)))
    g = isempty(R) ? ȳ : reduce_sum(ȳ, R)
    return (size(g) == insh ? g : apply(ReshapeOp(insh), g),)
end

# `srcs` are tensors whose axes can supply the RUNTIME SIZE of a dynamic target
# axis that the operand `x` doesn't itself carry (e.g. a bias/seed broadcast over
# a :B batch — the size lives in a sibling tensor). Matched by Dim identity.
function broadcast_to(x::AbstractTensor, shape, bdims; srcs::AbstractVector = AbstractTensor[])
    shape = Tuple(shape); bdims = collect(Int, bdims)
    (shape == size(x) && bdims == collect(1:ndims(x))) && return x       # no-op broadcast
    sa = size(x); n = length(shape)
    # Which dynamic target axes does `x` NOT supply itself? Those need a run-time
    # source (from `x` if it carries the dim, else from `srcs`).
    covered(j) = (i = findfirst(==(j), bdims); i !== nothing && !dims_equal(sa[i], 1))
    needs_dyn = any(j -> !(shape[j] isa Int) && !covered(j), 1:n)
    needs_dyn || return apply(BroadcastToOp(shape, bdims, Any[]), x)      # static path
    dynsize = Vector{Any}(undef, n)                                       # resolve every axis' size
    for j in 1:n
        if shape[j] isa Int
            dynsize[j] = Int(shape[j])
        elseif (i = findfirst(==(j), bdims); i !== nothing && !dims_equal(sa[i], 1))
            dynsize[j] = (x, i)                                           # x carries this dim
        else
            hit = nothing
            for s in srcs, a in 1:ndims(s)
                dims_equal(size(s)[a], shape[j]) && (hit = (s, a); break)
            end
            hit === nothing &&
                error("broadcast_to: no source supplies the runtime size of dynamic target axis $j " *
                      "($(shape[j])); pass a `srcs=` tensor that carries it.")
            dynsize[j] = hit
        end
    end
    return apply(BroadcastToOp(shape, bdims, dynsize), x)
end

# left-pad (NumPy) alignment: a rank-`ra` input maps to the TRAILING `ra` axes.
_trailing_bdims(ra, n) = collect((n - ra + 1):n)

function _bcast(f, a::AbstractTensor, b::AbstractTensor)
    s = broadcast_shapes(size(a), size(b))
    n = length(s)
    # Either operand may supply a dynamic axis the other lacks (e.g. bias over :B).
    return f(broadcast_to(a, s, _trailing_bdims(ndims(a), n); srcs = AbstractTensor[a, b]),
             broadcast_to(b, s, _trailing_bdims(ndims(b), n); srcs = AbstractTensor[a, b]))
end

# More specific than Base's generic `broadcasted`, so these intercept .+/.-/.*/./
# before Julia's array-broadcast path can engage. (`*` and `/` pass the STRICT
# elementwise ops `mul`/`/` as the base — the broadcast_to wrappers make the
# shapes equal first, so strictness is never violated.)
Base.broadcasted(::typeof(+), a::AbstractTensor, b::AbstractTensor) = _bcast(+, a, b)
Base.broadcasted(::typeof(-), a::AbstractTensor, b::AbstractTensor) = _bcast(-, a, b)
Base.broadcasted(::typeof(*), a::AbstractTensor, b::AbstractTensor) = _bcast(mul, a, b)
Base.broadcasted(::typeof(/), a::AbstractTensor, b::AbstractTensor) = _bcast(/, a, b)
_scalar(a::AbstractTensor{T}, x::Real) where {T} = Constant(fill(convert(T, x)))
Base.broadcasted(f::Union{typeof(+),typeof(-),typeof(*),typeof(/)}, a::AbstractTensor, x::Real) = Base.broadcasted(f, a, _scalar(a, x))
Base.broadcasted(f::Union{typeof(+),typeof(-),typeof(*),typeof(/)}, x::Real, a::AbstractTensor) = Base.broadcasted(f, _scalar(a, x), a)
# Unsupported dotted ops (.^) and Array operands still error — via Julia's
# generic path — rather than silently doing the wrong thing. Curated messages
# for those would need a dedicated tensor BroadcastStyle (deferred).

# --- ⊙ : the circled-dot Hadamard product — a spelled alias of `.*`
# (broadcasting elementwise multiply). ⊙ is the textbook elementwise-product
# glyph (LSTM/GRU gating, TensorCore.jl's ⊙=hadamard); it is NOT a dot product
# — that is `dot`/`⋅` (einsum.jl).
⊙(a::AbstractTensor, b::AbstractTensor) = _bcast(mul, a, b)
⊙(a::AbstractTensor, x::Real)           = _bcast(mul, a, _scalar(a, x))
⊙(x::Real, a::AbstractTensor)           = _bcast(mul, _scalar(a, x), a)

# --- scalar · tensor ergonomics. Bare `+ - * /` are strict same-shape for two
# tensors, but a Real scalar has no shape to mismatch — it broadcasts (rank-0
# stretches trivially), so `2f0 * x`, `x / 2f0`, etc. keep working.
Base.:*(x::Real, a::AbstractTensor) = _bcast(mul, _scalar(a, x), a)
Base.:*(a::AbstractTensor, x::Real) = _bcast(mul, a, _scalar(a, x))
Base.:/(a::AbstractTensor, x::Real) = _bcast(/, a, _scalar(a, x))
Base.:/(x::Real, a::AbstractTensor) = _bcast(/, _scalar(a, x), a)

# =====================================================================
# Concatenate / slice / pad — fixed-layout shape ops whose backward passes are
# ONE ANOTHER (concat→slice, slice→pad, pad→slice), so the trio is closed under
# differentiation to any order. `concatenate` of vec'd tensors is what packs a
# scope-selection's gradients into a single flat bundle (∇, gradients.jl).
#
# slice takes 1-based INCLUSIVE unit-stride ranges; pad takes per-axis low/high
# zero-widths (no interior). Strided slice / interior pad are a later extension;
# this subset {concat, unit-slice, edge-pad} stays closed on its own.
# =====================================================================

# --- concatenate along one axis -------------------------------------
struct ConcatOp <: Op
    dim::Int                                       # 1-based Julia axis
end
function outshape(op::ConcatOp, shapes...)
    n = length(first(shapes))
    all(s -> length(s) == n, shapes) ||
        error("concatenate: all inputs must share rank; got $(map(length, shapes))")
    (1 <= op.dim <= n) || error("concatenate: dim $(op.dim) out of range for rank $n")
    for d in 1:n, s in shapes[2:end]
        d == op.dim && continue
        dims_equal(first(shapes)[d], s[d]) ||
            error("concatenate: inputs differ off the concat axis at dim $d")
    end
    total = sum(s[op.dim] for s in shapes)
    return ntuple(d -> d == op.dim ? total : first(shapes)[d], n)
end
lower(op::ConcatOp, ins...) = shlo.concatenate(IR.Value[t.value for t in ins];
    result_0  = mlir_type(eltype(first(ins)), outshape(op, map(size, ins)...)),
    dimension = ir_attr("$(ndims(first(ins)) - op.dim) : i64"))
function vjp(op::ConcatOp, ȳ, ins, out)              # slice each input's slab back out
    off = 0; cts = AbstractTensor[]
    for t in ins
        len  = size(t, op.dim)
        rngs = ntuple(d -> d == op.dim ? (off+1:off+len) : (1:size(ȳ, d)), ndims(ȳ))
        push!(cts, slice(ȳ, rngs...)); off += len
    end
    return Tuple(cts)
end
concatenate(ins::AbstractTensor...; dim::Integer = 1) = apply(ConcatOp(Int(dim)), ins...)

# --- slice (1-based inclusive, unit stride) -------------------------
struct SliceOp <: Op
    starts::Vector{Int}
    stops::Vector{Int}
end
function outshape(op::SliceOp, sa)
    n = length(sa)
    (length(op.starts) == n && length(op.stops) == n) ||
        error("slice: need $n ranges for a rank-$n tensor")
    for d in 1:n
        (sa[d] isa Int) || error("slice: axis $d is symbolic; slice needs static extents")
        (1 <= op.starts[d] <= op.stops[d] <= sa[d]) ||
            error("slice: range $(op.starts[d]):$(op.stops[d]) out of bounds for axis $d (size $(sa[d]))")
    end
    return ntuple(d -> op.stops[d] - op.starts[d] + 1, n)
end
lower(op::SliceOp, a) = (rev = ndims(a):-1:1; shlo.slice(a.value;
    result_0      = mlir_type(eltype(a), outshape(op, size(a))),
    start_indices = i64array([op.starts[d] - 1 for d in rev]),
    limit_indices = i64array([op.stops[d]      for d in rev]),
    strides       = i64array([1 for _ in rev])))
function vjp(op::SliceOp, ȳ, ins, out)               # scatter ȳ back into a zero tensor (edge pad)
    insh = size(ins[1])
    return (pad(ȳ; low = [op.starts[d] - 1 for d in 1:ndims(ȳ)],
                   high = [insh[d] - op.stops[d] for d in 1:ndims(ȳ)]),)
end
slice(a::AbstractTensor, ranges::AbstractUnitRange...) =
    apply(SliceOp(collect(Int, first.(ranges)), collect(Int, last.(ranges))), a)

# --- pad (per-axis low/high zero edges; no interior) ----------------
struct PadOp <: Op
    low::Vector{Int}
    high::Vector{Int}
    value::Float64                                  # fill, cast to eltype at lower
end
function outshape(op::PadOp, sa)
    n = length(sa)
    (length(op.low) == n && length(op.high) == n) || error("pad: need $n (low,high) pairs")
    return ntuple(d -> op.low[d] + sa[d] + op.high[d], n)   # Int + Dim: Poly arithmetic handles symbolic
end
function lower(op::PadOp, a)
    T = eltype(a); rev = ndims(a):-1:1
    pv = shlo.constant(; value = IR.DenseElementsAttribute(fill(convert(T, op.value))), output = mlir_type(T, ()))
    push!(session().block, pv)
    return shlo.pad(a.value, IR.result(pv, 1);
        result_0         = mlir_type(T, outshape(op, size(a))),
        edge_padding_low  = i64array([op.low[d]  for d in rev]),
        edge_padding_high = i64array([op.high[d] for d in rev]),
        interior_padding  = i64array([0 for _ in rev]))
end
vjp(op::PadOp, ȳ, ins, out) =                        # slice out the original (unpadded) region
    (slice(ȳ, ntuple(d -> (op.low[d]+1 : op.low[d]+size(ins[1], d)), ndims(ȳ))...),)
pad(a::AbstractTensor, value::Real = zero(eltype(a)); low, high) =
    apply(PadOp(collect(Int, low), collect(Int, high), Float64(value)), a)

# --- axis_size: a scalar tensor holding the RUNTIME size of one axis ----------
# The keystone for reductions over a dynamic axis (mean over a :B batch): a
# static axis folds to a Constant; a symbolic one reads its size at run time via
# get_dimension_size and converts it to the tensor's element type. Non-
# differentiable (it reads the shape, not the data), so its vjp is `nothing`.
struct AxisSizeOp <: Op
    dim::Int
end
outshape(::AxisSizeOp, sa) = ()
function lower(op::AxisSizeOp, x)
    gd = shlo.get_dimension_size(x.value; dimension = ir_attr("$(ndims(x) - op.dim) : i64"))
    push!(session().block, gd)                        # rank-0 i32
    return shlo.convert(IR.result(gd, 1); result = mlir_type(eltype(x), ()))
end
vjp(::AxisSizeOp, ȳ, ins, out) = (nothing,)
function axis_size(x::AbstractTensor, d::Integer)
    d = Int(d)
    (1 <= d <= ndims(x)) || error("axis_size: axis $d out of range for a rank-$(ndims(x)) tensor")
    s = size(x, d)
    return s isa Int ? Constant(convert(eltype(x), s)) : apply(AxisSizeOp(d), x)
end
