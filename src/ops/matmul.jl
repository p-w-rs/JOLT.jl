# =====================================================================
# Matrix multiply — `*`, mirroring `@` in NumPy/JAX/TensorFlow.
#
# Batched: operands of EQUAL rank ≥ 2 with matching leading "batch" dims,
# contracting the last dim of A with the second-last of B:
#     (b…, m, k) · (b…, k, n)  ->  (b…, m, n)
# (plain 2-D is just the b…-empty case). The contracted dim and every batch dim
# must be provably equal — symbolic dims welcome, they lower to `?`. Backward
# swaps the last two dims (the batched transpose) and reuses batched `*`, so
# it's taped → higher order works:  Ā = ȳ · Bᵀ,  B̄ = Aᵀ · ȳ.
#
# Not yet: rank-mismatched forms like (b,l,d)·(d,k) and batch-dim broadcasting
# — both need the reduce/broadcast layer; reshape to 2-D for now.
# =====================================================================
struct MatMulOp <: Op end

function outshape(::MatMulOp, sa, sb)
    (length(sa) >= 2 && length(sa) == length(sb)) ||
        error("*: batched matmul needs two tensors of equal rank ≥ 2; got $sa · $sb " *
              "(reshape to 2-D for rank-mismatched contractions).")
    for i in 1:length(sa)-2
        dims_equal(sa[i], sb[i]) ||
            error("*: batch dim $i mismatch — $sa · $sb ($(sa[i]) vs $(sb[i]))")
    end
    dims_equal(sa[end], sb[end-1]) ||
        error("*: inner dims must match — $sa · $sb ($(sa[end]) vs $(sb[end-1])). " *
              "Declare same_dim! if they are equal.")
    return (sa[1:end-2]..., sa[end-1], sb[end])
end

# dot_general dimension numbers for rank R: batch dims 0…R-3, contract A[R-1]·B[R-2].
_dotdims(R::Int) = (b = join(0:R-3, ", ");
    "#stablehlo.dot<lhs_batching_dimensions = [$b], rhs_batching_dimensions = [$b], " *
    "lhs_contracting_dimensions = [$(R-1)], rhs_contracting_dimensions = [$(R-2)]>")

function lower(::MatMulOp, a, b)
    R = ndims(a)
    out = (size(a)[1:end-2]..., size(a, R-1), size(b, R))
    shlo.dot_general(a.value, b.value;
        result_0 = mlir_type(eltype(a), out),
        dot_dimension_numbers = ir_attr(_dotdims(R)))
end

# swap the last two dims, batch dims fixed — the batched transpose (Bᵀ).
_swaplast(t::Tensor) = (R = ndims(t); permutedims(t, (1:R-2..., R, R-1)))

vjp(::MatMulOp, ȳ, ins, out) = (ȳ * _swaplast(ins[2]), _swaplast(ins[1]) * ȳ)

Base.:*(a::Tensor, b::Tensor) = apply(MatMulOp(), a, b)
