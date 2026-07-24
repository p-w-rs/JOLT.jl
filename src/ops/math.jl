# =====================================================================
# Elementwise math & conditionals — unary activations, select, compare.
#
# The unary activations are shape-preserving and read their OWN forward output
# in the backward pass (d√x = ȳ/(2·out), d tanh = ȳ(1−out²), …), so `out` is
# threaded to vjp. Every backward is built from ordinary taped ops, so they
# differentiate to any order.
#
# `select`/`compare` are the runtime-branching pair. `compare` builds an i1
# predicate and is a NON-differentiable hard stop (its vjp is `nothing` for both
# inputs); `select` picks between two same-shape branches by that predicate and
# routes ȳ to the taken side. This is how a compiled graph branches on a runtime
# flag (train/inference) with no recompilation — see Flag / trainmode!.
# =====================================================================

# --- unary activations ----------------------------------------------
struct SqrtOp <: Op end
outshape(::SqrtOp, sa)     = sa
lower(::SqrtOp, a)         = shlo.sqrt(a.value)
vjp(::SqrtOp, ȳ, ins, out) = (ȳ / (2f0 * out),)                                # ȳ / (2√x)
Base.sqrt(a::AbstractTensor) = apply(SqrtOp(), a)

struct RsqrtOp <: Op end
outshape(::RsqrtOp, sa)     = sa
lower(::RsqrtOp, a)         = shlo.rsqrt(a.value)
vjp(::RsqrtOp, ȳ, ins, out) = (neg(0.5f0 * mul(ȳ, mul(out, mul(out, out)))),)  # -½·ȳ·x^{-3/2} = -½·ȳ·out³
rsqrt(a::AbstractTensor) = apply(RsqrtOp(), a)

struct TanhOp <: Op end
outshape(::TanhOp, sa)     = sa
lower(::TanhOp, a)         = shlo.tanh(a.value)
vjp(::TanhOp, ȳ, ins, out) = (mul(ȳ, 1f0 .- mul(out, out)),)                   # ȳ (1 − tanh²)
Base.tanh(a::AbstractTensor) = apply(TanhOp(), a)

struct SigmoidOp <: Op end
outshape(::SigmoidOp, sa)     = sa
lower(::SigmoidOp, a)         = shlo.logistic(a.value)
vjp(::SigmoidOp, ȳ, ins, out) = (mul(ȳ, mul(out, 1f0 .- out)),)                # ȳ·σ·(1−σ)
sigmoid(a::AbstractTensor) = apply(SigmoidOp(), a)

# --- select: pick per-element by an i1 predicate --------------------
# pred may be scalar (broadcasts over the branches) or the branches' shape; the
# two branches share shape+dtype. Backward routes ȳ to the taken side, 0 to the
# other; the predicate is non-differentiable.
struct SelectOp <: Op end
same_eltype(::SelectOp) = false                       # pred is i1; branches are floats
function outshape(::SelectOp, sp, sa, sb)
    _samedims("select", sa, sb)                       # the two branches must match
    isempty(sp) || _samedims("select predicate", sp, sa)
    return sa
end
lower(::SelectOp, p, a, b) =
    shlo.select(p.value, a.value, b.value; result = mlir_type(eltype(a), size(a)))
vjp(::SelectOp, ȳ, ins, out) = (nothing,
    apply(SelectOp(), ins[1], ȳ, zeros_like(ins[2])),
    apply(SelectOp(), ins[1], zeros_like(ins[3]), ȳ))
select(p::AbstractTensor, a::AbstractTensor, b::AbstractTensor) = apply(SelectOp(), p, a, b)

# --- compare: elementwise i1 predicate (NON-differentiable) ---------
struct CompareOp <: Op
    dir::String        # one of EQ NE GE GT LE LT
end
outshape(::CompareOp, sa, sb) = _samedims("compare", sa, sb)
lower(op::CompareOp, a, b) = shlo.compare(a.value, b.value;
    result_0 = mlir_type(Bool, size(a)),
    comparison_direction = ir_attr("#stablehlo<comparison_direction $(op.dir)>"))
vjp(::CompareOp, ȳ, ins, out) = (nothing, nothing)    # boolean output — a gradient hard stop
compare(a::AbstractTensor, b::AbstractTensor, dir::AbstractString) = apply(CompareOp(String(dir)), a, b)

# Dotted comparisons — `.> .>= .< .<= .== .!=` on tensors (and against a Real
# scalar), broadcasting like the arithmetic dotted ops. Each yields an i1 tensor.
_cmp(dir) = (a, b) -> apply(CompareOp(dir), a, b)
for (jlop, dir) in ((:(==), "EQ"), (:(!=), "NE"), (:(>=), "GE"), (:(>), "GT"), (:(<=), "LE"), (:(<), "LT"))
    @eval Base.broadcasted(::typeof($jlop), a::AbstractTensor, b::AbstractTensor) = _bcast(_cmp($dir), a, b)
    @eval Base.broadcasted(::typeof($jlop), a::AbstractTensor, x::Real) = _bcast(_cmp($dir), a, _scalar(a, x))
    @eval Base.broadcasted(::typeof($jlop), x::Real, a::AbstractTensor) = _bcast(_cmp($dir), _scalar(a, x), a)
end

# --- relu: sugar over select + compare (no new primitive) -----------
# max(x, 0): where x > 0 keep x, else 0. Its gradient falls out of select's.
relu(a::AbstractTensor) = select(a .> 0f0, a, zeros_like(a))
