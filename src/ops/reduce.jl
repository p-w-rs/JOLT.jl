# =====================================================================
# Reductions — sum over a set of axes.
#
# `reduce_sum` is the keystone the broadcasting layer needs: its backward pass
# is `broadcast_to`, and broadcast_to's backward pass is `reduce_sum` (they
# close the loop, so both are differentiable to any order). It also gives real
# scalar losses: `sum(x)` reduces every axis to a rank-0 scalar.
#
# stablehlo.reduce needs a reduction REGION — a nested block computing the
# combiner (here `add`). `_reduce_region` builds it by hand from the MLIR
# bindings (two rank-0 block args -> stablehlo.add -> stablehlo.return).
# =====================================================================

# The `+`-combiner region:  (a: tensor<T>, b: tensor<T>) { return a + b }.
function _reduce_region(::Type{T}) where {T}
    sc  = mlir_type(T, ())                              # rank-0 tensor<T>
    blk = IR.Block([sc, sc], [IR.Location(), IR.Location()])
    acc = shlo.add(IR.argument(blk, 1), IR.argument(blk, 2))
    push!(blk, acc)
    push!(blk, shlo.return_(IR.Value[IR.result(acc, 1)]))
    reg = IR.Region()
    push!(reg, blk)
    return reg
end

# reduce over `dims` (1-based axes), dropping them from the shape (rank falls).
struct ReduceSumOp <: Op
    dims::Vector{Int}
end
function outshape(op::ReduceSumOp, sa)
    all(d -> 1 <= d <= length(sa), op.dims) ||
        error("sum: reduce dims $(op.dims) out of range for a rank-$(length(sa)) tensor")
    return Tuple(sa[i] for i in eachindex(sa) if !(i in op.dims))
end
function lower(op::ReduceSumOp, x)
    T = eltype(x)
    init = shlo.constant(; value = IR.DenseElementsAttribute(fill(zero(T))), output = mlir_type(T, ()))
    push!(session().block, init)                       # the 0-identity, in the current block
    return shlo.reduce(
        IR.Value[x.value], IR.Value[IR.result(init, 1)];
        result_0   = IR.Type[mlir_type(T, outshape(op, size(x)))],
        dimensions = i64array(sort!([ndims(x) - d for d in op.dims])),   # Julia axis d → MLIR axis N-d
        body       = _reduce_region(T),
    )
end
# backward: replicate ȳ across the reduced axes — broadcast it into the input
# shape, mapping ȳ's dims to the kept axes (the reduced axes become new/replicated).
# `ins[1]` (the reduced input) is passed as the shape-source: if a reduced axis is
# dynamic (e.g. summing over a :B batch), its runtime size is read from ins[1] via
# dynamic_broadcast_in_dim — which is what makes gradients of dynamic-dim graphs work.
vjp(op::ReduceSumOp, ȳ, ins, out) =
    (broadcast_to(ȳ, size(ins[1]), [i for i in 1:ndims(ins[1]) if !(i in op.dims)]; srcs = AbstractTensor[ins[1]]),)

function reduce_sum(x::AbstractTensor, dims)
    ax = sort!(unique(Int[d for d in dims]))
    isempty(ax) && return x                              # reducing zero axes is the identity
    return apply(ReduceSumOp(ax), x)
end
# NOTE: reduce_sum DROPS the reduced axes (rank falls), unlike Base.sum(A; dims=)
# which keeps them as size-1. `sum(x)` (no dims) reduces everything to a scalar.
Base.sum(x::AbstractTensor; dims = Colon()) =
    reduce_sum(x, dims === Colon() ? (1:ndims(x)) : (dims isa Integer ? (dims,) : dims))
