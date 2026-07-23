# =====================================================================
# Tensors
#
# A tensor is an immutable SSA handle in one of four ROLES, each a concrete
# subtype of AbstractTensor{T,N} (T = element type, N = rank). The role IS the
# type: dispatch and `isa` replace any role tag, and each struct names exactly
# the payload it needs.
#
#   Argument — a runtime input (placeholder). MAY be symbolic, so its shape is
#              NTuple{N,Dim}. No payload.
#   Variable — a learnable value: concrete shape + an initializer (`init`). Its
#              array is realized at COMPILE time (not at graph-build time), so it
#              owns no host data here. Concrete Int dims only.
#   Constant — a literal value baked into the graph, created from a concrete
#              value (scalar or array). Carries that value (`data`). Concrete
#              Int dims only.
#   Result   — the output of an op. MAY be symbolic (its shape is propagated from
#              the inputs). No payload. Produced ONLY by the ops layer — it has
#              no public constructor, so a user can never build one directly.
#
# Only Argument and Result may be symbolic, and that rule lives in the field
# types (NTuple{N,Dim} vs NTuple{N,Int}), not a runtime check.
# =====================================================================
abstract type AbstractTensor{T,N} end

# Argument & Result may be symbolic — shape slots are Dim (Int or Poly).
struct Argument{T,N} <: AbstractTensor{T,N}
    value::IR.Value                 # SSA value this tensor denotes
    shape::NTuple{N,Dim}
end
struct Result{T,N} <: AbstractTensor{T,N}
    value::IR.Value
    shape::NTuple{N,Dim}
end

# Variable & Constant own concrete storage — shape slots are Int, so "no
# symbolic dims here" is guaranteed by the type.
struct Variable{T,N} <: AbstractTensor{T,N}
    value::IR.Value
    shape::NTuple{N,Int}
    init::Function                  # (rng, T, dims...) -> Array, realized at compile time
end
struct Constant{T,N} <: AbstractTensor{T,N}
    value::IR.Value
    shape::NTuple{N,Int}
    data::Array{T,N}                # the literal value, kept for getvalue
end

# User-facing construction shorthands ONLY — internally we always spell the full
# names Argument / Variable / Constant. Result has no shorthand and no public
# constructor.
const Arg   = Argument
const Var   = Variable
const Const = Constant

Base.ndims(::Type{<:AbstractTensor{T,N}}) where {T,N} = N
Base.ndims(t::AbstractTensor) = ndims(typeof(t))
Base.eltype(::Type{<:AbstractTensor{T,N}}) where {T,N} = T
Base.eltype(t::AbstractTensor) = eltype(typeof(t))

# =====================================================================
# Copy semantics
#
# A tensor is an immutable SSA handle, so `copy`/`deepcopy` alias it. There is
# no independent host state to clone: a Variable's array doesn't exist until
# compile, and a Constant's value is a graph literal. To get a second, separately
# initialized Variable, just construct another one.
# =====================================================================
Base.copy(t::AbstractTensor)     = t
Base.deepcopy(t::AbstractTensor) = t

# =====================================================================
# Shape & indexing
# =====================================================================
function Base.length(t::AbstractTensor)
    isstatic(t.shape) || error("length undefined: $t has symbolic dimension(s)")
    return prod(t.shape)          # scalar (N=0) -> prod(()) == 1
end
Base.size(t::AbstractTensor) = t.shape
# Match Base.size(::AbstractArray, d): a trailing/out-of-range dim reads as 1,
# rather than a bare BoundsError (d < 1 is still a programming error).
Base.size(t::AbstractTensor, d::Integer) = d <= ndims(t) ? t.shape[d] : 1
Base.getindex(::AbstractTensor, ::Any...) =
    error("a tensor is symbolic and holds no data; build a slice/gather op instead of indexing")

# Plug concrete sizes into a symbolic shape and evaluate it (debugging /
# model-building aid):  realize(t, :B => 32, :H => 240)  ->  (32, 3, 15, 15)
realize(t::AbstractTensor, subs::Pair{Symbol,<:Integer}...) =
    map(d -> realize(d, Dict{Symbol,Int}(subs...)), t.shape)

# =====================================================================
# Identity & hashing (by SSA value)
# =====================================================================
Base.:(==)(a::AbstractTensor, b::AbstractTensor) = a.value == b.value
Base.isequal(a::AbstractTensor, b::AbstractTensor) = a == b
Base.hash(t::AbstractTensor, h::UInt) = hash(t.value, h)

# =====================================================================
# Broadcasting: a tensor is its own broadcastable leaf (TF/numpy semantics are
# implemented in the ops layer, not via Base's array machinery).
# =====================================================================
Base.Broadcast.broadcastable(t::AbstractTensor) = t

# =====================================================================
# Display  — e.g. Argument{Float32,2}(2×3), Constant{Float32,0}(scalar)
# =====================================================================
function Base.show(io::IO, t::AbstractTensor{T,N}) where {T,N}
    dims = isempty(t.shape) ? "scalar" :
           join((string(d) for d in t.shape), "×")
    print(io, nameof(typeof(t)), "{$T,$N}(", dims, ")")
end
Base.show(io::IO, ::MIME"text/plain", t::AbstractTensor) = show(io, t)

# =====================================================================
# Host-value access — Constant only.
#
# A Constant's value is a graph literal we keep a copy of, so it is readable. A
# Variable's value does not exist until compile (it holds an initializer, not
# data), and Arguments/Results never carry host data — reading any non-Constant
# is an error with a clear message rather than a silent surprise. There is no
# setvalue!: Constants are literals and Variables are seeded by their initializer.
# =====================================================================
getvalue(t::Constant) = t.data
getvalue(t::Variable) =
    error("a Variable has no host value until compile; read it from the compiled `vars` container")
getvalue(t::AbstractTensor) =
    error("only Constant tensors expose a host value; a $(nameof(typeof(t))) does not")

# The initializer a Variable will be realized with (internal; the ops/compile
# layers reach for it, users generally don't).
initializer(t::Variable) = t.init

# =====================================================================
# Construction surface
#
# Create a tensor by calling its role — Argument / Variable / Constant, or the
# shorthands Arg / Var / Const. A Variable's general form is
#     Var([T], [init], dims...; name)
# where the element type T (a Real) and the initializer `init` (a Function,
# default Zeros()) are both OPTIONAL leading positionals; the other roles take an
# optional `name` too. The tensor is registered under a role-first path (see the
# naming rule in session.jl):
#     named     -> (:<role>s, scope..., :<name>)   e.g. (:variables, :default, :W)
#     anonymous -> (:<role>s, scope..., :_<n>)      e.g. (:variables, :default, :_1)
#
#   Arg(32, 4, 4)                     Argument, default dtype, static shape
#   Arg()                             scalar Argument
#   Arg(Float32, 8, :N)               Argument, explicit dtype, symbolic dim
#   Var(784, 128)                     Variable, default dtype + Zeros init
#   Var(GlorotUniform(), 784, 128; name="W")   Variable, init positional
#   Var(Float32)                      scalar Variable (Zeros init)
#   Var(Float32, Ones(), 8, 8)        dtype + init + dims
#   Const([1f0, 2f0, 3f0])            Constant, value baked into the graph
#   Const(3.0f0)                      scalar Constant
#   Arg(Float64, 8, 8)                explicit dtype
#
# Invalid attempts fail fast: symbolic dims on a Variable, a value handed to a
# Variable, or a shape handed to a Constant. Results have no constructor at all.
# (To fetch a tensor by name, see getArgument/getVariable/getConstant/getTensor.)
# =====================================================================

# --- helpers ---------------------------------------------------------
_materialize(a::AbstractArray) = collect(a)                              # dense, owned copy
_materialize(::Type{T}, a::AbstractArray) where {T} = collect(T, a)

_var_symbolic_err(dims) =
    error("a Variable owns concrete storage: its dims must be Int (only an Argument " *
          "may be symbolic). Got $(dims)")
_var_value_err() =
    error("a Variable is shape + init, not data — use `Var([init,] dims...)` " *
          "(init is an optional Function). For a literal value, make a Constant: `Const(value)`.")

# --- Argument (placeholder) — the only role a USER can make symbolic --
Argument(; name=nothing) = push_arg!((); name=name)   # scalar arg (0-arg beats the varargs tie)
Argument(dims::Union{Integer,Symbol,Poly}...; name=nothing) = push_arg!(map(todim, dims); name=name)
Argument(::Type{T}, dims::Union{Integer,Symbol,Poly}...; name=nothing) where {T<:Real} =
    push_arg!(map(todim, dims), T; name=name)

# --- Variable: Var([T], [init], dims...) — element type T and initializer
#     `init` (a Function, default Zeros()) are optional leading positionals.
Variable(dims::Integer...; name=nothing) =
    push_var!(Zeros(), map(Int, dims); name=name)
Variable(init::Function, dims::Integer...; name=nothing) =
    push_var!(init, map(Int, dims); name=name)
Variable(::Type{T}, dims::Integer...; name=nothing) where {T<:Real} =
    push_var!(Zeros(), map(Int, dims), T; name=name)
Variable(::Type{T}, init::Function, dims::Integer...; name=nothing) where {T<:Real} =
    push_var!(init, map(Int, dims), T; name=name)
# reject symbolic dims (only an Argument may be symbolic)
Variable(dims::Union{Integer,Symbol,Poly}...; kw...) = _var_symbolic_err(dims)
Variable(init::Function, dims::Union{Integer,Symbol,Poly}...; kw...) = _var_symbolic_err(dims)
Variable(::Type{T}, dims::Union{Integer,Symbol,Poly}...; kw...) where {T<:Real} = _var_symbolic_err(dims)
Variable(::Type{T}, init::Function, dims::Union{Integer,Symbol,Poly}...; kw...) where {T<:Real} = _var_symbolic_err(dims)
# reject a concrete value (that is a Constant now)
Variable(::AbstractArray; kw...) = _var_value_err()
Variable(::Type{<:Real}, ::AbstractArray; kw...) = _var_value_err()
Variable(::AbstractFloat; kw...) = _var_value_err()
Variable(::Type{<:Real}, ::AbstractFloat; kw...) = _var_value_err()

# --- Constant: MUST carry a value; a shape is an error --------------
Constant(data::AbstractArray{<:Real}; name=nothing) = push_constant!(_materialize(data); name=name)
Constant(::Type{T}, data::AbstractArray{<:Real}; name=nothing) where {T<:Real} = push_constant!(_materialize(T, data); name=name)
Constant(x::Real; name=nothing) = push_constant!(fill(x); name=name)
Constant(::Type{T}, x::Real; name=nothing) where {T<:Real} = push_constant!(fill(T(x)); name=name)
# value-less forms (the ≥2-integer methods need two fixed args so a lone scalar
# above stays a *value*, not a shape):
Constant(; name=nothing) =
    error("a Constant needs a value; e.g. Const([1f0,2f0]) or Const(3f0)")
Constant(d1::Integer, d2::Integer, drest::Integer...; name=nothing) =
    error("a Constant needs a value, not a shape $((d1, d2, drest...)); use Const(array) or Const(scalar)")
Constant(::Type{<:Real}, d1::Integer, d2::Integer, drest::Integer...; name=nothing) =
    error("a Constant needs a value, not a shape; use Const(array) or Const(scalar)")
