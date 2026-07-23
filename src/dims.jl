# =====================================================================
# THIS FILE IS A COMPLETE AI ARTIFACT AND I DO NOT FULLY UNDERSTAND IT
# Dims — symbolic tensor dimensions as integer polynomials.
#
# A dimension (`Dim`) is either a concrete `Int` or a `Poly`: a sum of
# monomials (products of atoms) with integer coefficients — the same normal
# form JAX uses for shape polymorphism (_DimExpr), with TVM's ModularSet and
# ConstIntBound sub-analyzers layered on top for divisibility and range
# reasoning.
#
#   atom      DimVar(:H)                          H        a named runtime dim
#             Opaque(fld, [H, 16])                ⌊H/16⌋   irreducible op application
#   monomial  Monomial(H => 2, W => 1)            H²·W     product of atoms
#   poly      Poly(H²W => 2, 𝟙 => -5)             2H²W - 5 sum of monomials
#
# Normalization keeps two invariants:
#   * a dim with no symbolic part is ALWAYS a plain Int (never a Poly), and
#   * zero coefficients / zero terms are pruned eagerly,
# so structural Dict equality decides "provably equal for every valuation":
# commutativity, associativity, distribution, like-term merging and
# cancellation all fall out of the representation. Equality is total and
# sound one way (JAX-style): `true` means provably equal; `false` means
# "not provably equal" — never "provably different".
#
# Constraints (a per-session `Facts` store) are consulted at COMPARISON time,
# never baked into stored dims, so they may be declared at any point during
# graph construction and stored shapes / `realize` / `show` keep their
# original form. Every declaration is also logged so `check_facts` can verify
# real input sizes against it at run time, before data reaches IREE.
# =====================================================================

# ---------------------------------------------------------------------
# Types.  Order matters: Opaque's args are `Dim`s, so Monomial/Poly/Dim
# come first (they only need the abstract Atom).
# ---------------------------------------------------------------------
abstract type Atom end

struct Monomial                        # product of atoms:  {H=>2, W=>1}  ≡  H²·W
    powers::Dict{Atom,Int}
end

struct Poly                            # sum of monomials:  {H²W=>2, 𝟙=>-5}  ≡  2H²W - 5
    terms::Dict{Monomial,Int}
end

const Dim = Union{Int,Poly}            # what sits in a shape slot

struct DimVar <: Atom                  # a named runtime dimension, e.g. :B
    name::Symbol
end

struct Opaque <: Atom                  # an op we can't reduce algebraically (fld, mod,
    op::Function                       # max, min). Inert for equality; evaluable in
    args::Vector{Dim}                  # `realize` because it remembers op + args.
end

const MONE = Monomial(Dict{Atom,Int}())   # MONE = "monomial one": the empty product ≡ 1

# ---------------------------------------------------------------------
# Equality & hashing.  All structural; Dict `==`/`hash` are order-independent
# and collapse duplicate keys, which is what makes `H+W == W+H`,
# `(a+b)+c == a+(b+c)`, and `H+H == 2H` hold with no rewrite rules at all.
# Opaque compares args as ordered vectors (fld/mod are not commutative;
# the commutative ops max/min sort their args at construction).
# ---------------------------------------------------------------------
Base.:(==)(a::DimVar, b::DimVar)     = a.name === b.name
Base.:(==)(a::Opaque, b::Opaque)     = a.op === b.op && a.args == b.args
Base.:(==)(a::Monomial, b::Monomial) = a.powers == b.powers
Base.:(==)(a::Poly, b::Poly)         = a.terms == b.terms

Base.hash(a::DimVar,   h::UInt) = hash(a.name, hash(:DimVar, h))
Base.hash(a::Opaque,   h::UInt) = hash(a.args, hash(a.op, hash(:Opaque, h)))
Base.hash(m::Monomial, h::UInt) = hash(m.powers, hash(:Monomial, h))
Base.hash(p::Poly,     h::UInt) = hash(p.terms, hash(:Poly, h))

# ---------------------------------------------------------------------
# Normalization & construction.
# ---------------------------------------------------------------------

# The single exit gate every arithmetic result passes through: prune zero
# coefficients, and collapse "nothing symbolic left" back to a plain Int.
# Callers always hand `poly` a freshly-built Dict, so mutating it is safe.
function poly(terms::Dict{Monomial,Int})
    filter!(((_, c),) -> c != 0, terms)
    isempty(terms) && return 0
    (length(terms) == 1 && haskey(terms, MONE)) && return terms[MONE]
    return Poly(terms)
end

# View any Dim as a term map so arithmetic can treat Int and Poly uniformly.
_terms(x::Int)  = iszero(x) ? Dict{Monomial,Int}() : Dict(MONE => x)
_terms(p::Poly) = p.terms

atom_poly(a::Atom) = Poly(Dict(Monomial(Dict{Atom,Int}(a => 1)) => 1))   # the Dim "1·a¹"

# Surface -> internal: integers stay Ints, Symbols become variables,
# Dims pass through. This is the only lifting point.
todim(x::Integer) = Int(x)
todim(s::Symbol)  = atom_poly(DimVar(s))
todim(d::Dim)     = d

# ---------------------------------------------------------------------
# Arithmetic.  Core methods are the concrete trio (Poly,Poly)/(Poly,Int)/
# (Int,Poly) — NOT (Poly,Dim) — so the generic `Integer` forwarders below
# can't create dispatch ambiguities. `Int op Int` stays on Base untouched.
# ---------------------------------------------------------------------
_add(a::Dim, b::Dim) = poly(mergewith(+, _terms(a), _terms(b)))          # like terms merge
_neg(x::Int)  = -x
_neg(p::Poly) = poly(Dict(m => -c for (m, c) in p.terms))
_scale(k::Int, p::Poly) = poly(Dict(m => k * c for (m, c) in p.terms))   # distribution

_monmul(a::Monomial, b::Monomial) = Monomial(mergewith(+, a.powers, b.powers))  # exponents add
function _mulp(a::Poly, b::Poly)                                         # full distribution
    t = Dict{Monomial,Int}()
    for (ma, ca) in a.terms, (mb, cb) in b.terms
        m = _monmul(ma, mb)
        t[m] = get(t, m, 0) + ca * cb
    end
    return poly(t)
end

Base.:+(a::Poly, b::Poly) = _add(a, b)
Base.:+(a::Poly, b::Int)  = _add(a, b)
Base.:+(a::Int,  b::Poly) = _add(a, b)
Base.:+(p::Poly)          = p

Base.:-(p::Poly)          = _neg(p)
Base.:-(a::Poly, b::Poly) = _add(a, _neg(b))
Base.:-(a::Poly, b::Int)  = _add(a, -b)
Base.:-(a::Int,  b::Poly) = _add(a, _neg(b))

Base.:*(a::Poly, b::Poly) = _mulp(a, b)
Base.:*(a::Poly, k::Int)  = _scale(k, a)
Base.:*(k::Int,  a::Poly) = _scale(k, a)

function Base.:^(p::Poly, k::Integer)
    k < 0 && error("negative power of a symbolic dimension: $p^$k")
    r::Dim = 1
    for _ in 1:k
        r = r * p
    end
    return r
end
# Negative literal powers (p^-1) route through inv before ^ can reject them.
Base.inv(p::Poly) = error("cannot invert a symbolic dimension: 1/($p) is not an integer dim")
# Symbolic exponents are not representable (a Dim is a polynomial) — curated error.
Base.:^(::Union{Integer,Poly}, p::Poly) =
    error("symbolic exponents are not supported: a dimension is a polynomial, and ^($p) is not")

# Symbolic dims have no build-time total order — that's what the bounds
# analyzer is for. Curated error instead of a bare `isless` MethodError.
Base.isless(a::Poly, b::Dim) = _noorder(a, b)
Base.isless(a::Int, b::Poly) = _noorder(a, b)
_noorder(a, b) = error("symbolic dimensions are not ordered at build time: " *
                       "$a < $b is unknowable. Use provably_ge / bound! instead.")

# ---------------------------------------------------------------------
# Floor-division & mod.  The one reduction rule (exact when the VARIABLE
# part is a multiple of k, from the identity ⌊(k·q + c)/k⌋ = q + ⌊c/k⌋):
#
#   fld(16h, 16)      -> h            fld(16h + 8w, 8) -> 2h + w
#   fld(16h - 5, 16)  -> h - 1        mod(16h + 20, 16) -> 4
#   fld(H, 16)        -> ⌊H/16⌋       (opaque: H isn't provably a multiple)
#
# Anything else (symbolic divisor, Int/expr numerator over a symbolic
# divisor) stays a fully opaque atom. Divisors must be positive — dims
# never divide by zero or a negative.
# ---------------------------------------------------------------------
constof(p::Poly)      = get(p.terms, MONE, 0)                 # the constant coefficient
_vardiv(p::Poly, k)   = all(((m, c),) -> m == MONE || c % k == 0, p.terms)
_exactdiv(p::Poly, k) = poly(Dict(m => c ÷ k for (m, c) in p.terms if m != MONE))

opaque(op, args...) = atom_poly(Opaque(op, Dim[args...]))

function Base.fld(p::Poly, k::Int)
    k > 0 || error("fld of a symbolic dim requires a positive divisor (got $k)")
    return _vardiv(p, k) ? _exactdiv(p, k) + fld(constof(p), k) : opaque(fld, p, k)
end
function Base.mod(p::Poly, k::Int)
    k > 0 || error("mod of a symbolic dim requires a positive modulus (got $k)")
    return _vardiv(p, k) ? mod(constof(p), k) : opaque(mod, p, k)
end
Base.fld(a::Poly, b::Poly) = a == b ? 1 : opaque(fld, a, b)   # d/d = 1 (dims ≥ 1); else opaque
Base.fld(a::Int,  b::Poly) = opaque(fld, a, b)
Base.mod(a::Poly, b::Poly) = a == b ? 0 : opaque(mod, a, b)   # d mod d = 0; else opaque
Base.mod(a::Int,  b::Poly) = opaque(mod, a, b)

# Dims are ≥ 1, so truncating and flooring division agree; alias for ÷ / %.
Base.div(a::Poly, b::Union{Int,Poly}) = fld(a, b)
Base.div(a::Int,  b::Poly)            = fld(a, b)
Base.rem(a::Poly, b::Union{Int,Poly}) = mod(a, b)
Base.rem(a::Int,  b::Poly)            = mod(a, b)

# ---------------------------------------------------------------------
# max / min.  Built opaque (construction is facts-blind and pure); resolved
# at comparison time by `canon` when bounds can prove an ordering, and
# computed exactly by `realize`. Args are sorted so max(H,W) == max(W,H).
# ---------------------------------------------------------------------
function _extremum(op, a::Dim, b::Dim)
    a == b && return a
    (a isa Int && b isa Int) && return op(a, b)
    x, y = repr(a) <= repr(b) ? (a, b) : (b, a)               # canonical (commutative) order
    return opaque(op, x, y)
end
Base.max(a::Poly, b::Poly) = _extremum(max, a, b)
Base.max(a::Poly, b::Int)  = _extremum(max, a, b)
Base.max(a::Int,  b::Poly) = _extremum(max, a, b)
Base.min(a::Poly, b::Poly) = _extremum(min, a, b)
Base.min(a::Poly, b::Int)  = _extremum(min, a, b)
Base.min(a::Int,  b::Poly) = _extremum(min, a, b)

# Generic Integer widths (Int32, UInt8, Bool, …) funnel into the Int cores.
for f in (:+, :-, :*, :fld, :mod, :div, :rem, :max, :min)
    @eval Base.$f(a::Poly, b::Integer) = Base.$f(a, Int(b))
    @eval Base.$f(a::Integer, b::Poly) = Base.$f(Int(a), b)
end

# ---------------------------------------------------------------------
# realize — substitute concrete sizes and evaluate. Opaque atoms are inert
# for equality but transparent here: they re-apply their op to realized args.
#
#   realize(fld(todim(:H),16) + 1, :H => 240)  ->  16
# ---------------------------------------------------------------------
realize(d::Int, env::AbstractDict{Symbol,<:Integer}) = d
realize(p::Poly, env::AbstractDict{Symbol,<:Integer}) =
    sum(c * prod(realize(a, env)^e for (a, e) in m.powers; init = 1)
        for (m, c) in p.terms; init = 0)
realize(a::DimVar, env::AbstractDict{Symbol,<:Integer}) =
    Int(get(env, a.name) do
        error("realize: no value provided for dimension variable $(a.name)")
    end)
realize(a::Opaque, env::AbstractDict{Symbol,<:Integer}) =
    a.op((realize(x, env) for x in a.args)...)
realize(d::Dim, subs::Pair{Symbol,<:Integer}...) = realize(d, Dict{Symbol,Int}(subs...))
# Loosely-typed envs (e.g. a bare Dict()) convert instead of MethodError-ing.
realize(d::Dim, env::AbstractDict) = realize(d, Dict{Symbol,Int}(env))

# ---------------------------------------------------------------------
# Display.  Terms print highest-degree first, then alphabetically, with
# signs folded into the separators:  2H²W + H - 5.
# ---------------------------------------------------------------------
Base.show(io::IO, a::DimVar) = print(io, a.name)

_argstr(d::Dim) = (d isa Poly && length(d.terms) > 1) ? "($d)" : "$d"
function Base.show(io::IO, a::Opaque)
    if a.op === fld && length(a.args) == 2
        print(io, "⌊", _argstr(a.args[1]), "/", _argstr(a.args[2]), "⌋")
    elseif a.op === mod && length(a.args) == 2
        print(io, "(", a.args[1], " mod ", a.args[2], ")")
    else
        print(io, a.op, "(", join(a.args, ", "), ")")
    end
end

function Base.show(io::IO, m::Monomial)
    isempty(m.powers) && return print(io, "1")
    parts = sort!(["$(a)" * (e == 1 ? "" : "^$e") for (a, e) in m.powers])
    print(io, join(parts, "·"))
end

_mdeg(m::Monomial) = sum(values(m.powers); init = 0)          # total degree, for sorting
function Base.show(io::IO, p::Poly)
    ts = sort!(collect(p.terms); by = ((m, _),) -> (-_mdeg(m), sprint(show, m)))
    for (i, (m, c)) in enumerate(ts)
        i == 1 ? (c < 0 && print(io, "-")) : print(io, c < 0 ? " - " : " + ")
        a = abs(c)
        if m == MONE
            print(io, a)
        else
            a != 1 && print(io, a)
            print(io, m)
        end
    end
end

# =====================================================================
# Facts — the per-session constraint store (à la JAX symbolic constraints /
# TVM's analyzer bindings). Never rewrites stored dims; queries consult it
# lazily, so constraints may be declared at ANY point during construction.
# Every declaration is logged in `decls` so `check_facts` can re-verify it
# against real input sizes at run time.
#
#   same_dim!(F, :B, :D)              declare two dims equal (any two Dims)
#   same_dim!(F, :N, todim(:B)*todim(:T))   ... including polynomials
#   pin!(F, :D, 1)                    fix a dim to a constant
#   divisible!(F, :H, 16)             declare 16 | H
#   bound!(F, :B, lo=2, hi=512)       declare a range (default: dims ≥ 1)
# =====================================================================
struct Facts
    eq    ::Dict{Symbol,Symbol}       # union-find: variable -> representative
    rules ::Dict{Atom,Dim}            # oriented rewrite rules: atom -> replacement
    modv  ::Dict{Symbol,Int}          # modv = "modular divisor": k where k | var
    lo    ::Dict{Symbol,Int}          # per-variable lower bounds (default 1)
    hi    ::Dict{Symbol,Int}          # per-variable upper bounds (default unbounded)
    quot  ::Dict{Symbol,Tuple{Symbol,Int}}  # quot = "quotient": minted H÷k -> (H, k) memo,
                                            # so the quotient inherits H's bounds ÷ k
    decls ::Vector{Any}               # declaration log, for run-time check_facts
end
Facts() = Facts(Dict{Symbol,Symbol}(), Dict{Atom,Dim}(), Dict{Symbol,Int}(),
                Dict{Symbol,Int}(), Dict{Symbol,Int}(),
                Dict{Symbol,Tuple{Symbol,Int}}(), Any[])

function rep(F::Facts, s::Symbol)                 # follow the union-find chain to the root
    while haskey(F.eq, s) && F.eq[s] !== s
        s = F.eq[s]
    end
    return s
end

# --- atom-occurrence helpers (used by rule orientation's occurs check;
#     they recurse into Opaque args, where a variable can hide) ----------
_dim_contains(d::Int, a::Atom)  = false
_dim_contains(p::Poly, a::Atom) = any(_mono_contains(m, a) for m in keys(p.terms))
_mono_contains(m::Monomial, a::Atom) = any(_atom_contains(k, a) for k in keys(m.powers))
_atom_contains(k::Atom, a::Atom) =
    k == a || (k isa Opaque && any(_dim_contains(x, a) for x in k.args))

# `d` is exactly the polynomial "1·var" -> that variable's name, else nothing.
function _lone_var(d::Dim)
    d isa Poly || return nothing
    length(d.terms) == 1 || return nothing
    m, c = only(d.terms)
    c == 1 || return nothing
    length(m.powers) == 1 || return nothing
    a, e = only(m.powers)
    return (e == 1 && a isa DimVar) ? a.name : nothing
end

_lonemono(a::Atom) = Monomial(Dict{Atom,Int}(a => 1))

# Divide EVERY coefficient (constant included) — used when solving a rule.
_divall(x::Int, c::Int)  = x ÷ c
_divall(p::Poly, c::Int) = poly(Dict(m => v ÷ c for (m, v) in p.terms))
_divok(x::Int, c::Int)   = x % c == 0
_divok(p::Poly, c::Int)  = all(v % c == 0 for v in values(p.terms))

# ---------------------------------------------------------------------
# same_dim! — declare that two dims are equal. Fast path: two plain
# variables merge in the union-find (migrating any facts keyed on the
# absorbed variable). General path: orient `canon(a-b) == 0` into a rewrite
# rule by solving for an isolated atom — preferring Opaque atoms, matching
# JAX's "replace floordiv(a,b) with c" — with an occurs check so a cyclic
# rule can never be created.
#
#   same_dim!(F, :N, todim(:B)*todim(:T))    -> rule  N => B·T
#   same_dim!(F, todim(:H)+todim(:W), 10)    -> rule  H => 10 - W
#   same_dim!(F, fld(todim(:H),16), :h)      -> rule  ⌊H/16⌋ => h
#   same_dim!(F, 2*todim(:H), 3*todim(:W))   -> error (no unit atom to solve for)
# ---------------------------------------------------------------------
function same_dim!(F::Facts, a::Union{Symbol,Integer,Poly}, b::Union{Symbol,Integer,Poly})
    da, db = todim(a), todim(b)
    va, vb = _lone_var(da), _lone_var(db)
    if va !== nothing && vb !== nothing                       # var ≡ var: union-find
        ra, rb = rep(F, va), rep(F, vb)
        ra === rb || _merge_var!(F, ra, rb)
    else
        d = canon(da - db, F)                                 # fold existing facts first
        if d !== 0
            d isa Int && error("unsatisfiable dimension constraint: $da == $db " *
                               "(difference simplifies to the constant $d)")
            _orient_rule!(F, d, da, db)
        end
    end
    # Log for the run-time gate only AFTER the constraint was accepted — a
    # rejected declaration must leave the store (and the log) untouched.
    push!(F.decls, (kind = :eq, lhs = da, rhs = db))
    return nothing
end

# Merge variable ra into rb, migrating rules / divisors / bounds keyed on ra.
# A rule carried by ra is NOT copied verbatim onto rb (its right-hand side may
# mention rb — a cycle — or clash with rb's own rule): it is re-declared through
# the same_dim! front door, so the occurs check and unsatisfiability detection
# fire exactly as they would for a fresh constraint (e.g. pin!(:B,3);
# pin!(:D,5); same_dim!(:B,:D) errors loudly). On failure the union and the
# popped rule are rolled back, leaving the store as it was.
function _merge_var!(F::Facts, ra::Symbol, rb::Symbol)
    ka = DimVar(ra)
    rule_a = haskey(F.rules, ka) ? pop!(F.rules, ka) : nothing
    F.eq[ra] = rb
    if rule_a !== nothing
        try
            same_dim!(F, rb, rule_a)              # re-orient: occurs check + contradiction
        catch
            delete!(F.eq, ra)                     # roll back to a consistent store
            F.rules[ka] = rule_a
            rethrow()
        end
    end
    haskey(F.modv, ra) && (F.modv[rb] = lcm(get(F.modv, rb, 1), pop!(F.modv, ra)))
    if haskey(F.lo, ra) || haskey(F.lo, rb)
        F.lo[rb] = max(get(F.lo, ra, 1), get(F.lo, rb, 1)); delete!(F.lo, ra)
    end
    if haskey(F.hi, ra) || haskey(F.hi, rb)
        F.hi[rb] = min(get(F.hi, ra, typemax(Int)), get(F.hi, rb, typemax(Int))); delete!(F.hi, ra)
    end
    return nothing
end

# Orient `d == 0` (d already canonicalized, ≠ constant) into `atom => rhs`.
function _orient_rule!(F::Facts, d::Poly, da, db)
    # Candidates: atoms appearing alone in a degree-1 monomial AND nowhere
    # else in d (including inside other atoms' Opaque args — solving for
    # those would immediately recreate themselves: a cycle).
    cands = Atom[]
    for (m, _) in d.terms
        (m == MONE || length(m.powers) != 1) && continue
        a, e = only(m.powers)
        e == 1 || continue
        elsewhere = any(m2 != m && _mono_contains(m2, a) for (m2, _) in d.terms)
        elsewhere || push!(cands, a)
    end
    # Prefer replacing Opaque atoms (JAX's direction: floordiv(a,b) => c),
    # then variables, deterministically by printed form.
    sort!(cands; by = a -> (a isa Opaque ? 0 : 1, sprint(show, a)))
    for a in cands
        c = d.terms[_lonemono(a)]
        rest = d - c * atom_poly(a)                          # d = c·a + rest
        _divok(rest, c) || continue                          # need c | every coeff of rest
        rhs = _divall(_neg(rest), c)                         # a == -rest / c
        _dim_contains(todim(rhs), a) && continue             # occurs check (cycle guard)
        F.rules[a] = rhs
        return nothing
    end
    error("cannot orient constraint $da == $db into a rewrite rule: no isolated " *
          "unit-coefficient atom to solve for. If this encodes divisibility " *
          "(e.g. ⌊H/k⌋·k == H), declare divisible!(:H, k) instead; otherwise " *
          "introduce a shared symbolic dim for both sides.")
end

# Sugar. pin! is just an equality with a constant; divisible! records a
# modular divisor (composing by lcm if declared twice); bound! tightens.
pin!(F::Facts, s::Union{Symbol,Poly}, v::Integer) = same_dim!(F, s, Int(v))

function divisible!(F::Facts, s::Symbol, k::Integer)
    k >= 2 || error("divisible!: modulus must be ≥ 2 (got $k)")
    r = rep(F, s)
    F.modv[r] = lcm(get(F.modv, r, 1), Int(k))
    push!(F.decls, (kind = :div, var = s, k = Int(k)))
    return nothing
end

function bound!(F::Facts, s::Symbol; lo::Integer = 1, hi::Union{Integer,Nothing} = nothing)
    r = rep(F, s)
    nlo = max(get(F.lo, r, 1), Int(lo))
    nhi = hi === nothing ? get(F.hi, r, typemax(Int)) : min(get(F.hi, r, typemax(Int)), Int(hi))
    nlo <= nhi || error("contradictory bounds for $s: lo=$nlo > hi=$nhi")
    F.lo[r] = nlo
    F.hi[r] = nhi
    push!(F.decls, (kind = :bound, var = s, lo = nlo, hi = nhi))
    return nothing
end

# ---------------------------------------------------------------------
# canon — rewrite a dim under the current facts, FOR COMPARISON ONLY.
# The result is never stored: shapes keep their original symbols. Runs to
# a fixpoint (rule chains like x => y+1, y => 2 need one pass each; the
# occurs check at insertion keeps the rule set acyclic, the cap is a
# belt-and-braces guard).
# ---------------------------------------------------------------------
const _CANON_ITERS = 32

canon(d::Int, ::Facts) = d
function canon(p::Poly, F::Facts)
    for _ in 1:_CANON_ITERS
        q = _canon_once(p, F)
        q isa Int && return q
        q == p && return q
        p = q
    end
    return p                                                  # conservative on cap
end

_canon_once(p::Poly, F::Facts) =
    sum(c * prod(catom(a, F)^e for (a, e) in m.powers; init = 1)
        for (m, c) in p.terms; init = 0)

function catom(v::DimVar, F::Facts)
    a = DimVar(rep(F, v.name))
    # KNOWN LIMITATION (conservative, never unsound): a rewrite rule shadows a
    # modv divisor on the same variable, and Opaque rule keys can go stale when
    # later facts change how their args canonicalize — in both cases a declared
    # fact merely becomes unprovable at build time; check_facts still enforces
    # it at run time.
    haskey(F.rules, a) && return F.rules[a]                   # equality/pin rewrite
    k = get(F.modv, a.name, 1)
    if k > 1                                                  # H -> k·(H÷k), exact
        q = Symbol(a.name, :÷, k)
        F.quot[q] = (a.name, k)                               # memo: q inherits H's bounds ÷ k
        return k * atom_poly(DimVar(q))
    end
    return atom_poly(a)
end

function catom(o::Opaque, F::Facts)
    args = Dim[canon(todim(x), F) for x in o.args]
    if length(args) == 2 && (o.op === max || o.op === min)    # bounds may resolve these
        provably_ge(args[1], args[2], F) && return o.op === max ? args[1] : args[2]
        provably_ge(args[2], args[1], F) && return o.op === max ? args[2] : args[1]
    end
    r = o.op(args...)                                         # re-apply: fld/mod may reduce
    if r isa Poly && length(r.terms) == 1                     # a rule on the rebuilt atom?
        m, c = only(r.terms)
        if c == 1 && length(m.powers) == 1
            a, e = only(m.powers)
            e == 1 && haskey(F.rules, a) && return F.rules[a]
        end
    end
    return r
end

# ---------------------------------------------------------------------
# ModularSet analyzer (TVM). Every dim's value lies in {coeff·t + base}:
# (0, b) is exactly b; (1, 0) is "anything". Propagated bottom-up, so
# divisibility of derived dims is PROVED, not rewritten:
#
#   divisible!(F, :A, 4); divisible!(F, :B, 4)
#   modset(A·B)  -> (16, 0)      so  provably_divisible(A·B, 16, F)
#   modset(A+B)  -> (4, 0)
#   modset(16h+3)-> (16, 3)      so  mod(16h+3, 16) is provably 3
# ---------------------------------------------------------------------
_ms_norm(c::Int, b::Int) = c == 0 ? (0, b) : (abs(c), mod(b, abs(c)))
_ms_add((c1, b1), (c2, b2)) = (g = gcd(c1, c2); g == 0 ? (0, b1 + b2) : _ms_norm(g, b1 + b2))
_ms_mul((c1, b1), (c2, b2)) =
    (c1 == 0 && c2 == 0) ? (0, b1 * b2) :
    _ms_norm(gcd(gcd(c1 * c2, c1 * b2), c2 * b1), b1 * b2)
_ms_scale(k::Int, ms) = _ms_mul((0, k), ms)

modset(d::Int, ::Facts)    = (0, d)
modset(v::DimVar, F::Facts) = (get(F.modv, rep(F, v.name), 1), 0)
modset(o::Opaque, ::Facts)  = (1, 0)                          # unknown residue
_ms_mono(m::Monomial, F::Facts) =
    reduce(_ms_mul, (reduce(_ms_mul, Iterators.repeated(modset(a, F), e); init = (0, 1))
                     for (a, e) in m.powers); init = (0, 1))
modset(p::Poly, F::Facts) =
    reduce(_ms_add, (_ms_scale(c, _ms_mono(m, F)) for (m, c) in p.terms))

function provably_divisible(d::Union{Symbol,Integer,Poly}, k::Integer, F::Facts)
    dd = canon(todim(d), F)
    dd isa Int && return dd % k == 0
    c, b = modset(dd, F)
    return c == 0 ? b % k == 0 : (c % k == 0 && b % k == 0)
end

# ---------------------------------------------------------------------
# ConstIntBound analyzer (TVM). Interval [lo, hi] per dim, propagated with
# saturating arithmetic (so unbounded × unbounded can't overflow). The
# default bound for every variable is [1, ∞): dimensions are at least 1
# (JAX's implicit constraint) — which lets e.g. max(H, 1) resolve to H
# with no declarations at all.
# ---------------------------------------------------------------------
const _INF = typemax(Int) >> 8                # ±_INF are INFINITY SENTINELS, not numbers:
                                              # any |x| ≥ _INF is treated as unbounded, and
                                              # the arithmetic below keeps them absorbing
                                              # (∞ + finite = ∞; ∞ + -∞ = indeterminate →
                                              # widen outward). Treating the sentinel as a
                                              # finite number is how bounds become unsound.
_sat(x::Int128) = Int(clamp(x, Int128(-_INF), Int128(_INF)))

# Addition: per endpoint, infinities absorb — and the indeterminate ∞ + (-∞)
# widens to the CONSERVATIVE side (lo -> -∞, hi -> +∞), never to a number.
function _cadd(a, b)
    lo = (a[1] <= -_INF || b[1] <= -_INF) ? -_INF :
         (a[1] >= _INF  || b[1] >= _INF)  ?  _INF : _sat(Int128(a[1]) + b[1])
    hi = (a[2] >= _INF  || b[2] >= _INF)  ?  _INF :
         (a[2] <= -_INF || b[2] <= -_INF) ? -_INF : _sat(Int128(a[2]) + b[2])
    return (lo, hi)
end
# Multiplication: Int128 corner arithmetic gets the sign logic of ±∞ right on
# its own (∞·0 = 0 included), and _sat re-widens overflow outward to ±∞.
function _cmul(a, b)
    corners = (Int128(a[1]) * b[1], Int128(a[1]) * b[2],
               Int128(a[2]) * b[1], Int128(a[2]) * b[2])
    return (_sat(minimum(corners)), _sat(maximum(corners)))
end
_cpow(b, e) = reduce(_cmul, Iterators.repeated(b, e); init = (1, 1))

# Flooring an endpoint must never divide the sentinel (⌊∞/k⌋ is still ∞).
_cfld(x::Int, k::Int) = x >= _INF ? _INF : x <= -_INF ? -_INF : fld(x, k)

cbound(d::Int, ::Facts) = (_sat(Int128(d)), _sat(Int128(d)))   # huge consts conflate with ∞: conservative
function cbound(v::DimVar, F::Facts)
    if haskey(F.quot, v.name)                     # minted quotient q = parent/k (exact):
        parent, k = F.quot[v.name]                # its range is the parent's, divided
        plo = get(F.lo, parent, 1)
        phi = min(get(F.hi, parent, _INF), _INF)
        return (max(1, cld(plo, k)), _cfld(phi, k))
    end
    r = rep(F, v.name)
    return (get(F.lo, r, 1), min(get(F.hi, r, _INF), _INF))
end
function cbound(o::Opaque, F::Facts)
    if o.op === fld && length(o.args) == 2 && o.args[2] isa Int && o.args[2] > 0
        b = cbound(o.args[1], F)
        return (_cfld(b[1], o.args[2]), _cfld(b[2], o.args[2]))
    elseif o.op === mod && length(o.args) == 2 && o.args[2] isa Int && o.args[2] > 0
        return (0, o.args[2] - 1)
    elseif (o.op === max || o.op === min) && length(o.args) == 2
        b1, b2 = cbound(o.args[1], F), cbound(o.args[2], F)
        return o.op === max ? (max(b1[1], b2[1]), max(b1[2], b2[2])) :
                              (min(b1[1], b2[1]), min(b1[2], b2[2]))
    end
    return (-_INF, _INF)                                      # unknown opaque: conservative
end
cbound(a::Atom, ::Facts) = (-_INF, _INF)
_cmono(m::Monomial, F::Facts) =
    reduce(_cmul, (_cpow(cbound(a, F), e) for (a, e) in m.powers); init = (1, 1))
cbound(p::Poly, F::Facts) =
    reduce(_cadd, (_cmul((c, c), _cmono(m, F)) for (m, c) in p.terms); init = (0, 0))

function provably_ge(a::Union{Symbol,Integer,Poly}, b::Union{Symbol,Integer,Poly}, F::Facts)
    d = canon(todim(a) - todim(b), F)
    return d isa Int ? d >= 0 : cbound(d, F)[1] >= 0
end

# ---------------------------------------------------------------------
# The queries every op routes through. STRICT, JAX/TVM-style: prove it, or
# reject at build time with a message that names the override. A symbolic
# dim is never assumed to be 1, and two unproven dims are never assumed
# equal — the builder must say so (and check_facts verifies they told the
# truth at run time).
# ---------------------------------------------------------------------
dims_equal(a::Dim, b::Dim, F::Facts) = a == b || canon(a - b, F) === 0

function broadcast_dim(a::Dim, b::Dim, F::Facts)
    a === 1 && return b                                       # a literal 1 stretches
    b === 1 && return a
    canon(a, F) === 1 && return b                             # provably 1 via facts
    canon(b, F) === 1 && return a
    dims_equal(a, b, F) && return a                           # keep the original form
    error("cannot broadcast dimensions $a and $b: not provably equal and neither " *
          "is provably 1. If they are equal, declare same_dim!($a, $b); if one is " *
          "a broadcast axis, pin!(…, 1).")
end

function contract_dim(a::Dim, b::Dim, F::Facts)
    dims_equal(a, b, F) ||
        error("cannot contract dimensions $a and $b: not provably equal. " *
              "Declare same_dim!($a, $b) if they are.")
    return nothing
end

# TF/NumPy/JAX broadcasting: LEFT-pad the lower-rank shape with 1s, then
# resolve each aligned pair.  (N, M, 10) ⊕ (10,)  ->  (N, M, 10)
function broadcast_shapes(sa::Tuple, sb::Tuple, F::Facts)
    n = max(length(sa), length(sb))
    pa = (ntuple(_ -> 1, n - length(sa))..., sa...)
    pb = (ntuple(_ -> 1, n - length(sb))..., sb...)
    return ntuple(i -> broadcast_dim(pa[i], pb[i], F), n)
end

# ---------------------------------------------------------------------
# check_facts — the run-time gate. `env` maps each dimension variable to
# the concrete size read off the real input arrays; every logged
# declaration is re-verified numerically, so an override the data violates
# fails HERE, in Julia with a real message — never inside the compiled
# IREE graph.
# ---------------------------------------------------------------------
function check_facts(env::AbstractDict{Symbol,<:Integer}, F::Facts)
    for d in F.decls
        if d.kind === :eq
            l, r = realize(d.lhs, env), realize(d.rhs, env)
            l == r || error("dimension constraint violated: declared $(d.lhs) == $(d.rhs), " *
                            "but the data gives $l vs $r")
        elseif d.kind === :div
            v = realize(DimVar(d.var), env)
            v % d.k == 0 || error("dimension constraint violated: declared $(d.k) | $(d.var), " *
                                  "but the data gives $(d.var) = $v")
        elseif d.kind === :bound
            v = realize(DimVar(d.var), env)
            d.lo <= v <= d.hi || error("dimension constraint violated: declared " *
                                       "$(d.lo) ≤ $(d.var) ≤ $(d.hi), but the data gives $v")
        end
    end
    return nothing
end
check_facts(env::AbstractDict, F::Facts) = check_facts(Dict{Symbol,Int}(env), F)

# ---------------------------------------------------------------------
# MLIR lowering & the StableHLO cross-check. Symbolic identity exists only
# on the Julia side: every Poly lowers to the anonymous dynamic size `?`.
# `mlir_ok` is the secondary check run in push_op! — our computed dim vs
# the size StableHLO inferred — flagging a genuine disagreement only when
# BOTH are concrete: `?` is compatible with anything symbolic.
# ---------------------------------------------------------------------
dim_to_mlir(d::Int) = d
dim_to_mlir(::Poly) = IR.dynsize()

mlir_ok(ours::Dim, sz) = !(ours isa Int) || IR.isdynsize(sz) || ours == Int(sz)

# A shape is static iff every dim is a concrete Int (no symbolic Poly dims).
isstatic(dims) = all(d -> d isa Int, dims)

# ---------------------------------------------------------------------
# Session-facts conveniences. `current_facts()` (defined in session.jl, the
# single point of session access) supplies the active session's Facts, so
# graph-building code and the REPL never pass the store explicitly.
# ---------------------------------------------------------------------
canon(d::Dim)                     = canon(d, current_facts())
dims_equal(a::Dim, b::Dim)        = dims_equal(a, b, current_facts())
broadcast_dim(a::Dim, b::Dim)     = broadcast_dim(a, b, current_facts())
contract_dim(a::Dim, b::Dim)      = contract_dim(a, b, current_facts())
broadcast_shapes(sa::Tuple, sb::Tuple) = broadcast_shapes(sa, sb, current_facts())
same_dim!(a, b)                   = same_dim!(current_facts(), a, b)
pin!(s, v::Integer)               = pin!(current_facts(), s, v)
divisible!(s::Symbol, k::Integer) = divisible!(current_facts(), s, k)
bound!(s::Symbol; kw...)          = bound!(current_facts(), s; kw...)
check_facts(env::AbstractDict)    = check_facts(env, current_facts())
provably_divisible(d, k::Integer) = provably_divisible(d, k, current_facts())
provably_ge(a, b)                 = provably_ge(a, b, current_facts())
