# =====================================================================
# Op machinery — the contract every primitive plugs into.
#
# Each op is a struct <: Op that defines three colocated aspects:
#   outshape(op, in_shapes...) -> shape   validate inputs (STRICT: prove or
#                                         reject at build time) + infer the
#                                         symbolic output shape.
#   lower(op, in_tensors...)   -> Operation   build the StableHLO op (reads
#                                             the tensors' `.value` operands).
#   vjp(op, ȳ, inputs, out)    -> cotangents   one per input, each built from
#                                             ordinary ops (backward = more
#                                             forward graph, so it's taped and
#                                             itself differentiable).
#
# `apply` wires them together and hands a finished OpNode to `push_op!`
# (session.jl). Op params that AREN'T tensors (a permutation, a target shape,
# a scale) live in the op struct; `apply` only threads tensor inputs.
# =====================================================================
abstract type Op end

function outshape end
function lower end
function vjp end

# Most ops require every input to share an element type. A few legitimately mix
# (e.g. `select`'s predicate is i1 while its branches are floats); those opt out.
same_eltype(::Op) = true

# One taped graph node: the primitive, its input tensors, the emitted StableHLO
# op, and the symbolic output shape dims.jl computed (the source of truth — MLIR
# only knows symbolic dims as anonymous `?`).
struct OpNode
    op::Op
    inputs::Vector{AbstractTensor}
    ir::IR.Operation
    shape::Tuple
end

function apply(op::Op, ins::AbstractTensor...)
    (!same_eltype(op) || allequal(eltype, ins)) ||
        error("$(typeof(op)): mixed element types $(map(eltype, ins))")
    shape = outshape(op, map(size, ins)...)
    ir    = lower(op, ins...)
    return push_op!(OpNode(op, collect(AbstractTensor, ins), ir, shape))
end

# --- MLIR attribute helpers (for ops carrying attributes) -----------
ir_attr(s::AbstractString) = parse(IR.Attribute, s)
i64array(dims) = isempty(dims) ? ir_attr("array<i64>") :
                 ir_attr("array<i64:$(join(dims, ","))>")
