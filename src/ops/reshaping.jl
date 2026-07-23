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
lower(op::TransposeOp, a)     = shlo.transpose(a.value; permutation = i64array(op.perm .- 1))
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
