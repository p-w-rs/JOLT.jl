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

Base.permutedims(a::Tensor, perm) = apply(TransposeOp(collect(Int, perm)), a)
Base.permutedims(a::Tensor)       = permutedims(a, reverse(1:ndims(a)))
Base.adjoint(a::Tensor)           = permutedims(a)          # real tensors: adjoint ≡ transpose

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

Base.reshape(a::Tensor, dims::Union{Integer,Symbol,Poly}...) =
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
end
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
    # A dynamic target axis needs a mapped, non-1 input dim to supply its runtime
    # size. Broadcasting INTO a dynamic new/stretched axis (e.g. a bias over a :B
    # batch) needs stablehlo.dynamic_broadcast_in_dim — not implemented yet, so we
    # reject at build time rather than emit unresolvable IR.
    for j in 1:n
        op.shape[j] isa Int && continue
        i = findfirst(==(j), op.bdims)
        (i !== nothing && !dims_equal(sa[i], 1)) ||
            error("broadcast_to: target axis $j is dynamic ($(op.shape[j])) with no input dim to supply " *
                  "its runtime size. Broadcasting into a dynamic new/stretched axis (e.g. a bias over a " *
                  ":B batch) needs dynamic_broadcast_in_dim (not implemented yet) — use a concrete size for now.")
    end
    return op.shape
end
lower(op::BroadcastToOp, x) = shlo.broadcast_in_dim(x.value;
    result_0 = mlir_type(eltype(x), op.shape),        # input dim i → output dim bdims[i], in reversed space
    broadcast_dimensions = i64array([length(op.shape) - b for b in reverse(op.bdims)]))
# backward: sum ȳ over the axes that were replicated (new) or stretched (input
# was size-1), then reshape back so the input's size-1 dims are restored.
function vjp(op::BroadcastToOp, ȳ, ins, out)
    insh = size(ins[1])
    new_ax     = [j for j in 1:length(op.shape) if !(j in op.bdims)]
    stretched  = [op.bdims[i] for i in eachindex(insh)
                  if dims_equal(insh[i], 1) && !dims_equal(op.shape[op.bdims[i]], 1)]
    R = sort!(unique(vcat(new_ax, stretched)))
    g = isempty(R) ? ȳ : reduce_sum(ȳ, R)
    return (size(g) == insh ? g : apply(ReshapeOp(insh), g),)
end

broadcast_to(x::Tensor, shape, bdims) =
    (Tuple(shape) == size(x) && collect(Int, bdims) == collect(1:ndims(x))) ? x :
    apply(BroadcastToOp(Tuple(shape), collect(Int, bdims)), x)

# left-pad (NumPy) alignment: a rank-`ra` input maps to the TRAILING `ra` axes.
_trailing_bdims(ra, n) = collect((n - ra + 1):n)

function _bcast(f, a::Tensor, b::Tensor)
    s = broadcast_shapes(size(a), size(b))
    n = length(s)
    return f(broadcast_to(a, s, _trailing_bdims(ndims(a), n)),
             broadcast_to(b, s, _trailing_bdims(ndims(b), n)))
end

# More specific than Base's generic `broadcasted`, so these intercept .+/.-/.*
# before Julia's array-broadcast path can engage.
Base.broadcasted(::typeof(+), a::Tensor, b::Tensor) = _bcast(+, a, b)
Base.broadcasted(::typeof(-), a::Tensor, b::Tensor) = _bcast(-, a, b)
Base.broadcasted(::typeof(*), a::Tensor, b::Tensor) = _bcast(mul, a, b)
_scalar(a::Tensor{T}, x::Real) where {T} = Tensor(Const, fill(convert(T, x)))
Base.broadcasted(f::Union{typeof(+),typeof(-),typeof(*)}, a::Tensor, x::Real) = Base.broadcasted(f, a, _scalar(a, x))
Base.broadcasted(f::Union{typeof(+),typeof(-),typeof(*)}, x::Real, a::Tensor) = Base.broadcasted(f, _scalar(a, x), a)
# Unsupported dotted ops (./ .^) and Array operands still error — via Julia's
# generic path — rather than silently doing the wrong thing. Curated messages
# for those would need a dedicated Tensor BroadcastStyle (deferred).
