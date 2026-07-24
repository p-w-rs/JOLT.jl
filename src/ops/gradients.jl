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
            ḡ === nothing && continue          # a non-differentiable input (e.g. select's predicate)
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

# A PackedGrad is a scope selection's gradients packed into ONE flat tensor plus
# the axes of the matching `vars` subtree. It is a first-class graph value — do
# `gs + v`, `sum(gs ⊙ gs)`, differentiate it again — and, once listed as a
# compile output, `fn` hands it back as a ComponentArray matching `vars.<subtree>`,
# so `Optimisers.update!(opt, vars.<subtree>, gs)` lines up by construction.
struct PackedGrad
    flat::AbstractTensor        # matched grads, vec'd and concatenated in the subtree's flat order
    axes                        # getaxes(template) — rebuilds the ComponentArray after fn
end
_flat(g::PackedGrad) = g.flat
_flat(x)             = x
for op in (:+, :-, :*, :/)      # whole-bundle arithmetic broadcasts through and keeps the axes
    @eval Base.$op(a::PackedGrad, b::PackedGrad)                = PackedGrad(broadcast($op, a.flat, b.flat), a.axes)
    @eval Base.$op(a::PackedGrad, b::Union{Real,AbstractTensor}) = PackedGrad(broadcast($op, a.flat, b), a.axes)
    @eval Base.$op(a::Union{Real,AbstractTensor}, b::PackedGrad) = PackedGrad(broadcast($op, a, b.flat), b.axes)
end
⊙(a::PackedGrad, b::PackedGrad) = PackedGrad(a.flat ⊙ b.flat, a.axes)
neg(g::PackedGrad)     = PackedGrad(neg(g.flat), g.axes)
Base.:-(g::PackedGrad) = neg(g)
Base.sum(g::PackedGrad) = sum(g.flat)          # a reduction → a plain scalar Result (drops the bundle)

# Pack every variable under `prefix` (a role-first path) into a PackedGrad shaped
# like the matching `vars` subtree — same nesting/order as _materialize_vars, so
# its axes equal getaxes(vars.<subtree>).
function _pack(s, prefix::Tuple, ct)
    L = length(prefix)
    L == 0 && error("∇: an empty wrt prefix selects nothing meaningful — name a role, e.g. :variables")
    matched = Tuple{Tuple,AbstractTensor}[]                 # (subtree-relative path, grad)
    for (p, t) in s.names
        _hasprefix(p, prefix) || continue
        rel = p[(L+1):end]
        isempty(rel) && error("∇: wrt=$prefix names a single tensor, not a scope — pass that tensor directly")
        push!(matched, (rel, _grad(ct, t)))
    end
    isempty(matched) && error("∇: wrt=$prefix matched no tensor. Prefixes start with a role; " *
        "present here: $(sort(unique(first(k) for k in keys(s.names))))")
    T = eltype(matched[1][2])
    template = ComponentArray(_nest([rel => zeros(T, size(g)...) for (rel, g) in matched]))
    base = UInt(pointer(getdata(template)))                 # sort by the ComponentArray's flat leaf order
    off(rel) = Int(UInt(pointer(foldl(getproperty, rel; init = template))) - base)
    order = sortperm([off(rel) for (rel, _) in matched])
    flats = [reshape(matched[i][2], prod(size(matched[i][2]))) for i in order]
    flat  = length(flats) == 1 ? flats[1] : concatenate(flats...)
    return PackedGrad(flat, getaxes(template))
end

# The return MIRRORS `wrt`: a tensor → its grad; a vector of tensors → an aligned
# Vector (duplicates allowed); a scope path (Symbol / Vector{Symbol}) → a
# PackedGrad; a Tuple of any of these → a Tuple of results aligned to the selectors.
_collect(s, t::AbstractTensor, ct)                    = _grad(ct, t)
_collect(s, ts::AbstractVector{<:AbstractTensor}, ct) = AbstractTensor[_grad(ct, t) for t in ts]
_collect(s, sym::Symbol, ct)                          = _pack(s, (sym,), ct)
_collect(s, v::AbstractVector{Symbol}, ct)            = _pack(s, Tuple(v), ct)
_collect(s, sels::Tuple, ct)                          = map(sel -> _collect(s, sel, ct), sels)

"""
    ∇(loss; wrt=:variables, seed=<ones>)   (also: gradient(loss; …))

Reverse-mode gradient of scalar `loss`. `wrt` selects what to differentiate
against and shapes the result:
  :variables              → everything under variables/     → a PackedGrad
  [:variables, :params]   → variables under variables/params → a PackedGrad matching vars.params
  W                       → that tensor                      → a single grad tensor
  [W, b, x]               → those tensors (dups ok, any role) → aligned Vector
  ([:variables,:params], x) → a Tuple of the above, aligned to the selectors

Scope paths are LITERAL and explicit: `:variables` is everything under
`variables/`; `[:variables, :params]` is `variables/params/`; `:params` alone
matches nothing. A PackedGrad packs to the matching `vars` subtree — list it as
a compile output and `fn` returns it as a ComponentArray ready for Optimisers.

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
