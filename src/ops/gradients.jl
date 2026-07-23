# =====================================================================
# Gradients — reverse-mode over the tape, plus gradient-routing ops.
#
# `∇` (aka `gradient`) seeds a scalar and walks the tape backward, accumulating
# a cotangent for every tensor it reaches. Every cotangent it builds is itself
# taped graph, so gradients are first-class tensors you can feed into more ops
# and differentiate again (second order, HVPs) with no special flag.
# =====================================================================

# A zero cotangent shaped like `x`: a constant when the shape is static, and
# `x - x` when it's symbolic (StableHLO can't build a `?`-shaped constant, but
# x−x is zero of exactly the right runtime shape).
zeros_like(x::AbstractTensor{T}) where {T} =
    isstatic(size(x)) ? Constant(zeros(T, size(x)...)) : x - x

# A constant-valued tensor shaped like `x` — DYNAMIC shapes included: it
# broadcasts a scalar into `x`'s (possibly `?`) shape via dynamic_broadcast_in_dim,
# reading the runtime sizes from `x` itself. This is how you seed a NON-scalar
# loss (whose shape may be dynamic): `∇(y; wrt=w, seed=ones_like(y))`.
fill_like(x::AbstractTensor, v) = broadcast_to(Constant(fill(convert(eltype(x), v))), size(x), Int[]; srcs = AbstractTensor[x])
ones_like(x::AbstractTensor)    = fill_like(x, one(eltype(x)))

# --- gradient-routing ops (identity forward, custom backward) -------
# Forward is a pass-through (optimization_barrier keeps it from being folded
# away); only the vjp differs.
_barrier(a::AbstractTensor) = shlo.optimization_barrier(IR.Value[a.value]; result = IR.Type[IR.type(a.value)])

# stop_gradient: cut the flow — ∂/∂input = 0 (nothing behind it gets gradient
# through this path). Used for detached targets (target nets, fixed labels).
struct StopGradient <: Op end
outshape(::StopGradient, sa)     = sa
lower(::StopGradient, a)         = _barrier(a)
vjp(::StopGradient, ȳ, ins, out) = (zeros_like(ins[1]),)
stop_gradient(a::AbstractTensor) = apply(StopGradient(), a)

# grad_reversal: identity forward, sign-flipped backward (domain-adversarial /
# DANN). A scale factor λ awaits the broadcast layer (scalar·tensor).
struct GradReversal <: Op end
outshape(::GradReversal, sa)     = sa
lower(::GradReversal, a)         = _barrier(a)
vjp(::GradReversal, ȳ, ins, out) = (neg(ȳ),)
grad_reversal(a::AbstractTensor) = apply(GradReversal(), a)

# --- reverse-mode engine --------------------------------------------
# `ct` maps each tensor to its accumulated ∂loss/∂tensor. The loop range is
# snapshotted before the walk BECAUSE the body appends to the tape (every vjp
# emits ops) — this pass differentiates only the ops that existed at entry; the
# new backward nodes sit above the snapshot and are (correctly) left for a
# later ∇ call to differentiate.
function _backprop(s::Session, loss::AbstractTensor, seed::AbstractTensor)
    ct = Dict{AbstractTensor,AbstractTensor}()
    ct[loss] = seed
    for i in lastindex(s.tape):-1:firstindex(s.tape)
        node, out = s.tape[i]
        haskey(ct, out) || continue
        ȳ = ct[out]
        for (x, ḡ) in zip(node.inputs, vjp(node.op, ȳ, node.inputs, out))
            # accumulate via the strict same-shape AddOp directly (not `+`), so a
            # future broadcasting `.+`/`+` can never silently un-strict this sum.
            ct[x] = haskey(ct, x) ? apply(AddOp(), ct[x], ḡ) : ḡ
        end
    end
    return ct
end

_ones_seed(loss::AbstractTensor) = isempty(size(loss)) ? Constant(one(eltype(loss))) :
    error("∇ needs a scalar loss (shape $(size(loss))); reduce it to a scalar (e.g. `sum`), " *
          "or pass an explicit cotangent — e.g. `seed = ones_like(loss)` (works for dynamic shapes too).")

# --- selection: which tensors to collect ----------------------------
_hasprefix(path::Tuple, pre::Tuple) = length(path) >= length(pre) && path[1:length(pre)] == pre
_grad(ct, t::AbstractTensor) = get(() -> zeros_like(t), ct, t)

# The return MIRRORS `wrt`: a tensor → a tensor; a vector of tensors → an
# aligned vector; a scope prefix (Symbol / Vector{Symbol}) → a path-keyed Dict.
_collect(s, t::AbstractTensor, ct)                    = _grad(ct, t)
_collect(s, ts::AbstractVector{<:AbstractTensor}, ct) = AbstractTensor[_grad(ct, t) for t in ts]
_collect(s, sym::Symbol, ct)                  = _collect(s, (sym,), ct)
_collect(s, v::AbstractVector{Symbol}, ct)    = _collect(s, Tuple(v), ct)
function _collect(s, pre::Tuple{Vararg{Symbol}}, ct)
    isempty(pre) && error("∇: an empty wrt prefix selects nothing meaningful — name a role, e.g. :variables")
    hits = [p => _grad(ct, t) for (p, t) in s.names if _hasprefix(p, pre)]
    isempty(hits) && error("∇: wrt=$pre matched no tensor. Prefixes start with a role; " *
        "present here: $(sort(unique(first(k) for k in keys(s.names))))")
    return Dict(hits)
end

"""
    ∇(loss; wrt=:variables, seed=<ones>)   (also: gradient(loss; …))

Reverse-mode gradient of scalar `loss`. `wrt` selects what to differentiate
against and shapes the result:
  :variables            → every variable            → Dict(path => grad)
  [:variables, :params] → variables under `params`  → Dict(path => grad)
  W                     → that tensor               → a single grad tensor
  [W, b]                → those tensors              → aligned Vector

Gradients are graph tensors — feed them into more ops and call `∇` again for
higher order. Pass `seed` (a cotangent shaped like `loss`) for a non-scalar
vector-Jacobian product.
"""
function ∇(loss::AbstractTensor; wrt = :variables, seed::AbstractTensor = _ones_seed(loss))
    s = session()
    _owned(s, loss) || error("∇: `loss` belongs to a different session than the active one")
    _samedims("∇ seed", size(seed), size(loss))    # a seed is a cotangent of loss — same shape
    ct = _backprop(s, loss, seed)
    return _collect(s, wrt, ct)
end
const gradient = ∇
