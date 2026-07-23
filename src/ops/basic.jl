# =====================================================================
# Basic elementwise primitives — simple forward, simple backward.
#
# All are SAME-SHAPE ONLY (StableHLO's add/multiply/divide/… don't broadcast);
# the broadcasting surface (`.+ .- .* ./`) lives in reshaping.jl, which wraps
# each operand in a broadcast_to and then reuses these strict ops. The bare
# operators `+ - * /` are ALL elementwise and require provably equal shapes —
# contraction (matmul, dot, einsum) has its own names (matmul.jl, einsum.jl).
# =====================================================================

# Strict same-shape rule shared by the binary elementwise ops.
function _samedims(name::String, sa::Tuple, sb::Tuple)
    length(sa) == length(sb) ||
        error("$name: rank mismatch — $(length(sa)) vs $(length(sb)) ($sa vs $sb)")
    for i in eachindex(sa)
        dims_equal(sa[i], sb[i]) ||
            error("$name: shapes $sa and $sb differ at dim $i ($(sa[i]) vs $(sb[i])). " *
                  "Bare $name requires equal shapes; declare same_dim! if they are equal.")
    end
    return sa
end

# --- add (elementwise):  d(a+b) = (ȳ, ȳ) ------------------------------------------
struct AddOp <: Op end
outshape(::AddOp, sa, sb) = _samedims("+", sa, sb)
lower(::AddOp, a, b)      = shlo.add(a.value, b.value)
vjp(::AddOp, ȳ, ins, out) = (ȳ, ȳ)
Base.:+(a::AbstractTensor, b::AbstractTensor) = apply(AddOp(), a, b)

# --- subtract (elementwise):  d(a-b) = (ȳ, -ȳ) ------------------------------------
struct SubOp <: Op end
outshape(::SubOp, sa, sb) = _samedims("-", sa, sb)
lower(::SubOp, a, b)      = shlo.subtract(a.value, b.value)
vjp(::SubOp, ȳ, ins, out) = (ȳ, neg(ȳ))
Base.:-(a::AbstractTensor, b::AbstractTensor) = apply(SubOp(), a, b)

# --- multiply (elementwise):  d(a·b) = (ȳ·b, ȳ·a) -------------------
# Each input's cotangent reads the OTHER forward input — why tape nodes keep
# their inputs. `*` is ELEMENTWISE (matmul is `matmul`/`⊡`, matmul.jl); `mul`
# is the spelled-out alias so backward passes read clearly.
struct MulOp <: Op end
outshape(::MulOp, sa, sb) = _samedims("*", sa, sb)
lower(::MulOp, a, b)      = shlo.multiply(a.value, b.value)
vjp(::MulOp, ȳ, ins, out) = (mul(ȳ, ins[2]), mul(ȳ, ins[1]))
mul(a::AbstractTensor, b::AbstractTensor) = apply(MulOp(), a, b)
Base.:*(a::AbstractTensor, b::AbstractTensor) = mul(a, b)

# --- negate (unary):  d(-a) = -ȳ ------------------------------------
struct NegOp <: Op end
outshape(::NegOp, sa)     = sa
lower(::NegOp, a)         = shlo.negate(a.value)
vjp(::NegOp, ȳ, ins, out) = (neg(ȳ),)
neg(a::AbstractTensor) = apply(NegOp(), a)
Base.:-(a::AbstractTensor) = neg(a)

# --- divide (elementwise):  z = a/b ⇒ (ȳ/b, -ȳ·a/b²) ----------------
# The one new StableHLO op the elementwise refactor introduces. Backward is
# built from divide + multiply + negate — every op it needs is itself taped, so
# the quotient rule differentiates to any order (higher-order closes).
struct DivOp <: Op end
outshape(::DivOp, sa, sb) = _samedims("/", sa, sb)
lower(::DivOp, a, b)      = shlo.divide(a.value, b.value)
vjp(::DivOp, ȳ, ins, out) = (apply(DivOp(), ȳ, ins[2]),                    # ∂a = ȳ / b
    neg(apply(DivOp(), mul(ȳ, ins[1]), mul(ins[2], ins[2]))))             # ∂b = -ȳ·a / b²
Base.:/(a::AbstractTensor, b::AbstractTensor) = apply(DivOp(), a, b)
