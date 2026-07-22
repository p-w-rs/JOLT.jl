# =====================================================================
# Ops — every operation is a struct subtyping `Op`, with its three aspects
# defined together in one place:
#
#   outshape(op, shapes...) -> shape      Dim-algebra rule: validates the
#                                         inputs (STRICT: prove or reject at
#                                         build time) and computes the
#                                         symbolic output shape.
#   lower(op, tensors...)   -> Operation  builds the StableHLO op from the
#                                         input Tensors (their .value fields
#                                         are the SSA operands).
#   vjp(op, ȳ, inputs, out) -> cotangents one per input, each built from
#                                         ordinary ops (so the backward pass
#                                         is just more forward graph).
#
# The generic `apply` wires them together and hands a finished OpNode to
# `push_op!` (session.jl) — ops.jl itself never touches the session; the
# only ambient state it reads is `current_facts()` via the dims queries.
# =====================================================================
abstract type Op end

# One taped graph node: everything the session (and later the gradient
# driver) needs to replay this op — which primitive, which input tensors,
# the emitted StableHLO op, and the symbolic output shape that dims.jl
# computed (the source of truth; MLIR's `?` can't carry symbols).
struct OpNode
    op::Op
    inputs::Vector{Tensor}
    ir::IR.Operation
    shape::Tuple
end

function apply(op::Op, ins::Tensor...)
    T = eltype(first(ins))
    all(eltype(t) === T for t in ins) ||
        error("$(typeof(op)): mixed element types $(map(eltype, ins))")
    shape = outshape(op, map(size, ins)...)
    ir    = lower(op, ins...)
    return push_op!(OpNode(op, collect(ins), ir, shape))
end

# =====================================================================
# Attribute helpers (for future ops that carry MLIR attributes).
# =====================================================================
# Parse an MLIR attribute from its textual form, e.g. "array<i64: 0, 1>".
ir_attr(s::AbstractString) = parse(IR.Attribute, s)
i64array(dims) =
    isempty(dims) ?
    ir_attr("array<i64>") :
    ir_attr("array<i64:$(join(dims, ","))>")

# =====================================================================
# Shared shape rule: strict same-shape elementwise.
#
# Bare `+` (like matrix addition) requires provably-equal shapes — no
# broadcasting. The broadcasting surface (`.+` etc.) arrives with the
# broadcast_to primitive and will route through broadcast_shapes instead.
# =====================================================================
function _samedims(opname::String, sa::Tuple, sb::Tuple)
    length(sa) == length(sb) ||
        error("$opname: rank mismatch — $(length(sa)) vs $(length(sb)) ($sa vs $sb)")
    for i in eachindex(sa)
        dims_equal(sa[i], sb[i]) ||
            error("$opname: shapes $sa and $sb differ at dim $i " *
                  "($(sa[i]) vs $(sb[i])). Bare $opname requires equal shapes; " *
                  "declare same_dim! if these dims are in fact equal.")
    end
    return sa
end

# =====================================================================
# add — elementwise addition, SAME SHAPE ONLY (stablehlo.add itself never
# broadcasts). The broadcasting surface `.+` will left-pad + broadcast_to
# both operands to the common shape and then reuse this exact op.
# d(a+b) = (ȳ, ȳ).
# =====================================================================
struct AddOp <: Op end
outshape(::AddOp, sa::Tuple, sb::Tuple) = _samedims("+", sa, sb)
lower(::AddOp, a::Tensor, b::Tensor) = shlo.add(a.value, b.value)
vjp(::AddOp, ȳ::Tensor, ins, out) = (ȳ, ȳ)

Base.:+(a::Tensor, b::Tensor) = apply(AddOp(), a, b)

# =====================================================================
# mul — elementwise multiplication, SAME SHAPE ONLY (like add: broadcasting
# is `.*`'s job once the broadcast layer lands; bare `*` stays reserved for
# matmul). d(a·b) = (ȳ·b, ȳ·a): each input's cotangent reads the *other*
# forward input — exactly why tape nodes store their inputs.
# =====================================================================
struct MulOp <: Op end
outshape(::MulOp, sa::Tuple, sb::Tuple) = _samedims("mul", sa, sb)
lower(::MulOp, a::Tensor, b::Tensor) = shlo.multiply(a.value, b.value)
vjp(::MulOp, ȳ::Tensor, ins, out) = (mul(ȳ, ins[2]), mul(ȳ, ins[1]))

mul(a::Tensor, b::Tensor) = apply(MulOp(), a, b)
