# =====================================================================
# Roles
#
# A tensor is exactly one of four kinds, and the kind is a type parameter so it
# drives dispatch (and hard-blocks invalid construction — see below):
#
#   Argument — a runtime input (placeholder). The ONLY role that may carry
#              symbolic / polynomial dims.
#   Variable — a learnable value: shape + an initializer. Its array is realized
#              at COMPILE time (not at graph-build time), so it owns no host data
#              here. Concrete Int dims only.
#   Constant — a literal value baked into the graph. Created by handing over a
#              concrete value (scalar or array). Concrete shape only.
#   Result   — the output of an op. Produced ONLY by the ops layer; a user can
#              never construct one directly.
# =====================================================================
@enum TensorRole Argument Variable Constant Result
const Arg   = Argument
const Var   = Variable
const Const = Constant
const Res   = Result

# =====================================================================
# Hard fallback default role. (The default *dtype* is a Session field, seeded
# to Float32 in Session(); change it at runtime with default_dtype!.)
# =====================================================================
const DEFAULT_ROLE = Argument

# =====================================================================
# Tensor
# T - element type (a Real), N - rank, R - TensorRole (a value parameter)
#
# The handle carries a role-specific `payload`:
#   Variable -> its initializer, a function (rng, T, dims...) -> Array
#   Constant -> its literal value, an Array{T,N}
#   Argument -> nothing
#   Result   -> nothing
# Because the four roles are disjoint, one union field says it all, and every
# accessor keys off the role (R) so there is never a "which variant is it?" check.
# =====================================================================
abstract type AbstractTensor{T,N,R} end

Base.ndims(::Type{<:AbstractTensor{T,N,R}}) where {T,N,R} = N
Base.ndims(t::AbstractTensor) = ndims(typeof(t))
Base.eltype(::Type{<:AbstractTensor{T,N,R}}) where {T,N,R} = T
Base.eltype(t::AbstractTensor) = eltype(typeof(t))

struct Tensor{T,N,R} <: AbstractTensor{T,N,R}
    value::IR.Value                         # SSA value this tensor denotes
    shape::NTuple{N,Dim}                    # Int dims are static; Poly dims are symbolic
    payload::Union{Nothing,Function,Array{T,N}}  # init fn (Var) | value (Const) | nothing
end

# Pull the role (an enum value) back out of the type, in the ndims/eltype style.
roleof(::Type{<:AbstractTensor{T,N,R}}) where {T,N,R} = R
roleof(t::AbstractTensor) = roleof(typeof(t))

# =====================================================================
# Copy semantics
#
# A Tensor is an immutable SSA handle, so `copy`/`deepcopy` alias it. There is
# no independent host state to clone: a Variable's value doesn't exist until
# compile, and a Constant's value is a graph literal. To get a second, separately
# initialized Variable, just construct another one.
# =====================================================================
Base.copy(t::Tensor)     = t
Base.deepcopy(t::Tensor) = t

# =====================================================================
# Shape & indexing
# =====================================================================
function Base.length(t::Tensor)
    isstatic(t.shape) || error("length undefined: $t has symbolic dimension(s)")
    return prod(t.shape)          # scalar (N=0) -> prod(()) == 1
end
Base.size(t::Tensor) = t.shape
# Match Base.size(::AbstractArray, d): a trailing/out-of-range dim reads as 1,
# rather than a bare BoundsError (d < 1 is still a programming error).
Base.size(t::Tensor, d::Integer) = d <= ndims(t) ? t.shape[d] : 1
Base.getindex(::Tensor, ::Any...) =
    error("Tensor is symbolic and holds no data; build a slice/gather op instead of indexing")

# Plug concrete sizes into a symbolic shape and evaluate it (debugging /
# model-building aid):  realize(t, :B => 32, :H => 240)  ->  (32, 3, 15, 15)
realize(t::Tensor, subs::Pair{Symbol,<:Integer}...) =
    map(d -> realize(d, Dict{Symbol,Int}(subs...)), t.shape)

# =====================================================================
# Identity & hashing (by SSA value)
# =====================================================================
Base.:(==)(a::Tensor, b::Tensor) = a.value == b.value
Base.isequal(a::Tensor, b::Tensor) = a == b
Base.hash(t::Tensor, h::UInt) = hash(t.value, h)

# =====================================================================
# Broadcasting: `Tensor` is its own broadcastable leaf (TF/numpy semantics
# are implemented in the ops layer, not via Base's array machinery).
# =====================================================================
Base.Broadcast.broadcastable(t::Tensor) = t

# =====================================================================
# Display
# =====================================================================
function Base.show(io::IO, t::Tensor{T,N,R}) where {T,N,R}
    dims = isempty(t.shape) ? "scalar" :
           join((string(d) for d in t.shape), "×")
    print(io, "Tensor{$T,$N,$R}($dims)")
end
Base.show(io::IO, ::MIME"text/plain", t::Tensor) = show(io, t)

# =====================================================================
# Host-value access — Constant only.
#
# A Constant's value is a graph literal we keep a copy of, so it is readable.
# A Variable's value does not exist until compile (it holds an initializer, not
# data), and Arguments/Results never carry host data — reading any of them is an
# error with a clear message rather than a silent surprise. There is no
# setvalue!: Constants are literals and Variables are seeded by their initializer.
# =====================================================================
getvalue(t::Tensor{T,N,Constant}) where {T,N} = t.payload::Array{T,N}
getvalue(t::Tensor{T,N,Variable}) where {T,N} =
    error("a Variable has no host value until compile; read it from the compiled `vars` container")
getvalue(t::Tensor) =
    error("only Constant tensors expose a host value; $(roleof(t)) does not")

# The initializer a Variable will be realized with (internal; the ops/compile
# layers reach for it, users generally don't).
initializer(t::Tensor{T,N,Variable}) where {T,N} = t.payload::Function

# =====================================================================
# Construction surface
#
# One rule of thumb decides the role when you don't name one:
#     INTEGERS / SYMBOLS describe a SHAPE   -> Argument
#     a FLOAT or an ARRAY describes a VALUE  -> Constant
# and Variables are always spelled out with `Var` (shape + `init`).
#
# Every constructor takes an optional `name`; Variables also take `init`
# (default `Zeros()`). The tensor is registered under a role-first path
# (see the naming rule in session.jl):
#     named    -> (:<role>s, scope..., :<name>)   e.g. (:variables, :default, :W)
#     anonymous-> (:<role>s, scope..., :_<n>)      e.g. (:variables, :default, :_1)
#
#   Tensor(32, 4, 4)                     Argument, default dtype, static shape
#   Tensor()                             scalar Argument
#   Tensor(Float32, 8, :N)               Argument, explicit dtype, symbolic dim
#   Tensor(Var, 784, 128; init=GlorotUniform(), name="W")   Variable
#   Tensor(Var, Float32)                 scalar Variable (Zeros init)
#   Tensor(Var, 10; init=Fill(0.1))      Variable, custom initializer
#   Tensor(Const, [1f0, 2f0, 3f0])       Constant, value baked into the graph
#   Tensor(3.0f0)                        scalar Constant
#   Tensor(Float64, Arg, 8, 8)           dtype + role prefix
#   Tensor("variables", "Dense", "W")    look a tensor up by its full path
#
# Invalid attempts fail fast with a message: symbolic dims on a Variable, a value
# handed to a Variable, a shape handed to a Constant, or constructing a Result.
# =====================================================================

# --- helpers ---------------------------------------------------------
_materialize(a::AbstractArray) = collect(a)                              # dense, owned copy
_materialize(::Type{T}, a::AbstractArray) where {T} = collect(T, a)

_var_symbolic_err(dims) =
    error("a Variable owns concrete storage: its dims must be Int (only an Argument " *
          "may be symbolic). Got $(dims)")
_var_value_err() =
    error("a Variable is shape + init, not data — use `Tensor(Var, dims...; init=…)`. " *
          "For a literal value, make a Constant: `Tensor(Const, value)`.")

# --- role / dtype prefixes trampoline runtime values onto Val dispatch,
#     forwarding ALL keywords (name, init, …) through.
Tensor(R::TensorRole, args...; kw...) = Tensor(Val(R), args...; kw...)
Tensor(::Type{T}, R::TensorRole, args...; kw...) where {T<:Real} = Tensor(Val(R), T, args...; kw...)

# --- Arguments (placeholders) — the only role that may be symbolic ---
Tensor(; name=nothing) = push_arg!((); name=name)   # scalar arg (breaks the empty-varargs tie)
Tensor(dims::Union{Integer,Symbol,Poly}...; name=nothing) = push_arg!(map(todim, dims); name=name)
Tensor(::Type{T}, dims::Union{Integer,Symbol,Poly}...; name=nothing) where {T<:Real} = push_arg!(map(todim, dims), T; name=name)
Tensor(::Val{Argument}, dims::Union{Integer,Symbol,Poly}...; name=nothing) = push_arg!(map(todim, dims); name=name)
Tensor(::Val{Argument}, ::Type{T}, dims::Union{Integer,Symbol,Poly}...; name=nothing) where {T<:Real} = push_arg!(map(todim, dims), T; name=name)

# --- Variable: concrete shape + an initializer (default Zeros()) -----
Tensor(::Val{Variable}, dims::Integer...; init=Zeros(), name=nothing) =
    push_var!(init, map(Int, dims); name=name)
Tensor(::Val{Variable}, ::Type{T}, dims::Integer...; init=Zeros(), name=nothing) where {T<:Real} =
    push_var!(init, map(Int, dims), T; name=name)
# reject symbolic dims (only an Argument may be symbolic)
Tensor(::Val{Variable}, dims::Union{Integer,Symbol,Poly}...; kw...) = _var_symbolic_err(dims)
Tensor(::Val{Variable}, ::Type{T}, dims::Union{Integer,Symbol,Poly}...; kw...) where {T<:Real} = _var_symbolic_err(dims)
# reject a concrete value (that is a Constant now)
Tensor(::Val{Variable}, ::AbstractArray; kw...) = _var_value_err()
Tensor(::Val{Variable}, ::Type{<:Real}, ::AbstractArray; kw...) = _var_value_err()
Tensor(::Val{Variable}, ::AbstractFloat; kw...) = _var_value_err()
Tensor(::Val{Variable}, ::Type{<:Real}, ::AbstractFloat; kw...) = _var_value_err()

# --- Constant: MUST carry a value; a shape is an error --------------
Tensor(::Val{Constant}, data::AbstractArray{<:Real}; name=nothing) = push_constant!(_materialize(data); name=name)
Tensor(::Val{Constant}, ::Type{T}, data::AbstractArray{<:Real}; name=nothing) where {T<:Real} = push_constant!(_materialize(T, data); name=name)
Tensor(::Val{Constant}, x::Real; name=nothing) = push_constant!(fill(x); name=name)
Tensor(::Val{Constant}, ::Type{T}, x::Real; name=nothing) where {T<:Real} = push_constant!(fill(T(x)); name=name)
# value-less forms (the ≥2-integer methods need two fixed args so a lone scalar
# above stays a *value*, not a shape):
Tensor(::Val{Constant}; name=nothing) =
    error("a Constant needs a value; e.g. Tensor(Const, [1f0,2f0]) or Tensor(Const, 3f0)")
Tensor(::Val{Constant}, d1::Integer, d2::Integer, drest::Integer...; name=nothing) =
    error("a Constant needs a value, not a shape $((d1, d2, drest...)); use Tensor(Const, array) or Tensor(Const, scalar)")
Tensor(::Val{Constant}, ::Type{<:Real}, d1::Integer, d2::Integer, drest::Integer...; name=nothing) =
    error("a Constant needs a value, not a shape; use Tensor(Const, array) or Tensor(Const, scalar)")

# --- bare value (no role word) => Constant ---------------------------
Tensor(data::AbstractArray{<:Real}; name=nothing) = push_constant!(_materialize(data); name=name)
Tensor(x::AbstractFloat; name=nothing) = push_constant!(fill(x); name=name)

# --- Result: never user-constructed ---------------------------------
Tensor(::Val{Result}, args...; kw...) =
    error("Result tensors are produced by operations, not constructed directly")

# --- lookup by full (role-first) path -------------------------------
Tensor(path::AbstractString...) = lookup(Symbol.(path))
