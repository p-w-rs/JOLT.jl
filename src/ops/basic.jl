# =====================================================================
# Basic elementwise primitives — simple forward, simple backward.
#
# All are SAME-SHAPE ONLY (StableHLO's add/multiply/… don't broadcast); the
# broadcasting surface (`.+`, `.*`) arrives later with a broadcast_to op and
# will reuse these. Bare `+`/`-`/`*`(elementwise via `mul`) require provably
# equal shapes.
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
Base.:+(a::Tensor, b::Tensor) = apply(AddOp(), a, b)

# --- subtract (elementwise):  d(a-b) = (ȳ, -ȳ) ------------------------------------
struct SubOp <: Op end
outshape(::SubOp, sa, sb) = _samedims("-", sa, sb)
lower(::SubOp, a, b)      = shlo.subtract(a.value, b.value)
vjp(::SubOp, ȳ, ins, out) = (ȳ, neg(ȳ))
Base.:-(a::Tensor, b::Tensor) = apply(SubOp(), a, b)

# --- multiply (elementwise):  d(a·b) = (ȳ·b, ȳ·a) -------------------
# Each input's cotangent reads the OTHER forward input — why tape nodes keep
# their inputs. `*` is reserved for matmul (matmul.jl), so this is `mul`.
struct MulOp <: Op end
outshape(::MulOp, sa, sb) = _samedims("mul", sa, sb)
lower(::MulOp, a, b)      = shlo.multiply(a.value, b.value)
vjp(::MulOp, ȳ, ins, out) = (mul(ȳ, ins[2]), mul(ȳ, ins[1]))
mul(a::Tensor, b::Tensor) = apply(MulOp(), a, b)

# --- negate (unary):  d(-a) = -ȳ ------------------------------------
struct NegOp <: Op end
outshape(::NegOp, sa)     = sa
lower(::NegOp, a)         = shlo.negate(a.value)
vjp(::NegOp, ȳ, ins, out) = (neg(ȳ),)
neg(a::Tensor) = apply(NegOp(), a)
Base.:-(a::Tensor) = neg(a)
