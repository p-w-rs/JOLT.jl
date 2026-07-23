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

# In reversed-dim space A_mlir=(k,m,rev_b…), B_mlir=(n,k,rev_b…). dot_general puts
# batch dims FIRST, so we swap operands (lhs=B, rhs=A) to get n before m, and for
# R>2 transpose the batch dims from the front to the back — dot_general → (rev_b…,
# n, m), then reorder to (n, m, rev_b…) = reversed(b…, m, n).
function lower(::MatMulOp, a, b)
    R = ndims(a); T = eltype(a)
    batch = join(2:R-1, ", ")                       # MLIR batch axes (empty for R=2)
    dn = "#stablehlo.dot<lhs_batching_dimensions = [$batch], rhs_batching_dimensions = [$batch], " *
         "lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>"
    m = size(a, R-1); n = size(b, R); revb = reverse(size(a)[1:end-2])
    dot = shlo.dot_general(b.value, a.value;        # swapped: lhs = B, rhs = A
        result_0 = IR.TensorType(Int[dim_to_mlir(d) for d in (revb..., n, m)], IR.Type(T)),
        dot_dimension_numbers = ir_attr(dn))
    R == 2 && return dot                             # (n, m) = reversed (m, n): the only op
    push!(session().block, dot)                      # batched: dot is auxiliary — append it here,
    return shlo.transpose(IR.result(dot, 1);         # then return the reorder (rev_b…,n,m) → (n,m,rev_b…)
        result = mlir_type(T, (size(a)[1:end-2]..., m, n)),
        permutation = i64array([R - 2, R - 1, (0:R-3)...]))
end

# swap the last two dims, batch dims fixed — the batched transpose (Bᵀ).
_swaplast(t::Tensor) = (R = ndims(t); permutedims(t, (1:R-2..., R, R-1)))

vjp(::MatMulOp, ȳ, ins, out) = (ȳ * _swaplast(ins[2]), _swaplast(ins[1]) * ȳ)

Base.:*(a::Tensor, b::Tensor) = apply(MatMulOp(), a, b)
