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
# Broadcasting — .+ / .- / .*  (NumPy/TF/JAX semantics).
#
# `broadcast_to` (stablehlo.broadcast_in_dim) stretches an operand to a target
# shape: input dim i maps to target dim `bdims[i]`; target dims NOT in `bdims`
# are new axes the operand is replicated across. `.+`/`.-`/`.*` broadcast BOTH
# operands to the common shape (dims.jl's broadcast_shapes) then reuse the
# strict same-shape op — so their gradients come for free from broadcast_to's
# vjp (a reduce_sum over the stretched/new axes) composed with the base op.
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

# More specific than Base's generic `broadcasted`, so these intercept .+/.-/.*
# before Julia's array-broadcast path can engage.
Base.broadcasted(::typeof(+), a::AbstractTensor, b::AbstractTensor) = _bcast(+, a, b)
Base.broadcasted(::typeof(-), a::AbstractTensor, b::AbstractTensor) = _bcast(-, a, b)
Base.broadcasted(::typeof(*), a::AbstractTensor, b::AbstractTensor) = _bcast(mul, a, b)
_scalar(a::AbstractTensor{T}, x::Real) where {T} = Constant(fill(convert(T, x)))
Base.broadcasted(f::Union{typeof(+),typeof(-),typeof(*)}, a::AbstractTensor, x::Real) = Base.broadcasted(f, a, _scalar(a, x))
Base.broadcasted(f::Union{typeof(+),typeof(-),typeof(*)}, x::Real, a::AbstractTensor) = Base.broadcasted(f, _scalar(a, x), a)
# Unsupported dotted ops (./ .^) and Array operands still error — via Julia's
# generic path — rather than silently doing the wrong thing. Curated messages
# for those would need a dedicated tensor BroadcastStyle (deferred).
