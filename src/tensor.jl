# =====================================================================
# Roles
# =====================================================================
@enum TensorRole Argument Variable Constant Result
const Arg   = Argument
const Var   = Variable
const Const = Constant
const Res   = Result

# =====================================================================
# Hard fallback defaults. The *session* default dtype (settable at runtime)
# lives in session.jl; this constant is only the seed value for it.
# =====================================================================
const DEFAULT_ELTYPE = Float32
const DEFAULT_ROLE   = Argument

# =====================================================================
# Shared mutable data cell.
#
# A Variable's host-side value lives in one of these boxes. Because
# `copy(t) === t` (a Tensor is just a handle), every alias of a variable
# points at the SAME box: updating through one handle is visible through all
# of them, and any number of call sites can read it. Arguments, Constants,
# and Results carry no box (their `data` field is `nothing`).
# =====================================================================
mutable struct DataBox{T,N}
    array::Array{T,N}
end

# =====================================================================
# Tensor
# T - element type (a Real), N - rank, R - TensorRole (a value parameter)
# =====================================================================
abstract type AbstractTensor{T,N,R} end

Base.ndims(::Type{<:AbstractTensor{T,N,R}}) where {T,N,R} = N
Base.ndims(t::AbstractTensor) = ndims(typeof(t))
Base.eltype(::Type{<:AbstractTensor{T,N,R}}) where {T,N,R} = T
Base.eltype(t::AbstractTensor) = eltype(typeof(t))

struct Tensor{T,N,R} <: AbstractTensor{T,N,R}
    value::IR.Value                    # SSA value this tensor denotes
    shape::NTuple{N,Dim}               # Int dims are static; Poly dims are runtime (symbolic)
    data::Union{Nothing,DataBox{T,N}}  # populated only for Variables
end

# Pull the role (an enum value) back out of the type, in the ndims/eltype style.
roleof(::Type{<:AbstractTensor{T,N,R}}) where {T,N,R} = R
roleof(t::AbstractTensor) = roleof(typeof(t))

# =====================================================================
# Copy semantics
# =====================================================================
# `copy` and `deepcopy` both alias: a Tensor is an immutable SSA handle, and a
# Variable's DataBox is meant to be shared. To get an *independent* variable
# (a fresh block argument seeded with a copy of the current data), use
# `duplicate` — that intentionally emits new IR.
Base.copy(t::Tensor)     = t
Base.deepcopy(t::Tensor) = t

function duplicate(t::Tensor{T,N,Variable}) where {T,N}
    return push_var!(copy(t.data.array))
end
duplicate(t::Tensor) =
    error("duplicate is only defined for Variable tensors (got $(roleof(t)))")

# =====================================================================
# Shape & indexing
# =====================================================================
function Base.length(t::Tensor)
    all(d -> d isa Int, t.shape) || error("length undefined: $t has symbolic dimension(s)")
    return prod(t.shape)          # scalar (N=0) -> prod(()) == 1
end
Base.size(t::Tensor) = t.shape
Base.size(t::Tensor, d::Integer) = t.shape[d]
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
# Host-value access — Variable only.
#
# Enforcement: only a Variable owns host data, so only a Variable can be read
# or updated. Every other role errors with a clear message rather than silently
# doing nothing.
# =====================================================================
function setvalue!(t::Tensor{T,N,Variable}, x::AbstractArray) where {T,N}
    size(x) == t.shape ||
        error("shape mismatch: variable is $(t.shape), got $(size(x))")
    t.data.array = collect(T, x)
    return t
end
setvalue!(t::Tensor{T,0,Variable}, x::Real) where {T} = (t.data.array = fill(T(x)); t)
setvalue!(t::Tensor{T,N,Variable}, x) where {T,N} =
    error("cannot set a Variable from a $(typeof(x)); provide an AbstractArray " *
          "(or a Real for a scalar Variable)")
setvalue!(t::Tensor, ::Any) =
    error("cannot set value: $(roleof(t)) tensors are not updatable — only Variable is")

getvalue(t::Tensor{T,N,Variable}) where {T,N} = t.data.array
getvalue(t::Tensor) =
    error("cannot read host value: $(roleof(t)) tensors hold no host data — only Variable does")

# =====================================================================
# Construction surface
#
# Every constructor accepts an optional `name` keyword. Whether or not you give
# one, the tensor is registered in the session under the CURRENT namespace:
#   - with `name="W"`  -> registered as (current_scope..., :W)
#   - without a name    -> registered as (current_scope..., :<role>#<n>)
# So `name` and the anonymous fallback are handled by the exact same path.
#
# Rule of thumb: INTEGERS describe a shape, FLOATS / ARRAYS describe a value.
#   Tensor(32, 4, 4)                     Argument, default dtype, static shape
#   Tensor()                             scalar Argument
#   Tensor(Float32, 8, :N)               Argument, explicit dtype, dynamic (symbolic) dim
#   Tensor(Var, 784, 128, name="W")      Variable, zero-initialized, named
#   Tensor(Var, Float32)                 scalar Variable, zero-initialized
#   Tensor(Var, randn(Float32, 3))       Variable, initialized from data
#   Tensor(Const, [1f0, 2f0, 3f0])       Constant, value embedded in the graph
#   Tensor(3.0f0)                        scalar Constant
#   Tensor(Float32, Arg, 8, 8)           dtype + role prefix
#   Tensor("Dense", "W")                 look up a previously-registered tensor
#
# A Constant MUST be given a value (there is deliberately no shape-only form);
# a Variable may be given just a shape (zero-init) or explicit data.
# =====================================================================

# --- helpers ---------------------------------------------------------
_materialize(a::AbstractArray) = collect(a)                              # dense, owned copy
_materialize(::Type{T}, a::AbstractArray) where {T} = collect(T, a)

# --- Arguments (placeholders) ---------------------------------------
Tensor(; name=nothing) = push_arg!(default_dtype(), (); name=name)   # scalar arg (also breaks the empty-varargs tie)
Tensor(dims::Union{Integer, Symbol, Poly}...; name=nothing) = push_arg!(default_dtype(), map(todim, dims); name=name)
Tensor(::Type{T}, dims::Union{Integer, Symbol, Poly}...; name=nothing) where {T<:Real} = push_arg!(T, map(todim, dims); name=name)

# --- Argument (explicit) --------------------------------------------
Tensor(::Val{Argument}, dims::Union{Integer, Symbol, Poly}...; name=nothing) = push_arg!(default_dtype(), map(todim, dims); name=name)
Tensor(::Val{Argument}, ::Type{T}, dims::Union{Integer, Symbol, Poly}...; name=nothing) where {T<:Real} = push_arg!(T, map(todim, dims); name=name)

# --- role / dtype prefixes trampoline runtime values onto Val dispatch,
#     forwarding the name keyword through.
Tensor(R::TensorRole, args...; name=nothing) = Tensor(Val(R), args...; name=name)
Tensor(::Type{T}, R::TensorRole, args...; name=nothing) where {T<:Real} = Tensor(Val(R), T, args...; name=name)

# --- Variable: shape -> zeros (scalar when no dims), or explicit data
Tensor(::Val{Variable}, dims::Integer...; name=nothing) = push_var!(zeros(default_dtype(), dims...); name=name)
Tensor(::Val{Variable}, ::Type{T}, dims::Integer...; name=nothing) where {T<:Real} = push_var!(zeros(T, dims...); name=name)
Tensor(::Val{Variable}, init::AbstractArray{<:Real}; name=nothing) = push_var!(_materialize(init); name=name)
Tensor(::Val{Variable}, ::Type{T}, init::AbstractArray{<:Real}; name=nothing) where {T<:Real} = push_var!(_materialize(T, init); name=name)
Tensor(::Val{Variable}, x::Real; name=nothing) = push_var!(fill(x); name=name)
Tensor(::Val{Variable}, ::Type{T}, x::Real; name=nothing) where {T<:Real} = push_var!(fill(T(x)); name=name)

# --- Constant: MUST carry a value -----------------------------------
Tensor(::Val{Constant}, data::AbstractArray{<:Real}; name=nothing) = push_constant!(_materialize(data); name=name)
Tensor(::Val{Constant}, ::Type{T}, data::AbstractArray{<:Real}; name=nothing) where {T<:Real} = push_constant!(_materialize(T, data); name=name)
Tensor(::Val{Constant}, x::Real; name=nothing) = push_constant!(fill(x); name=name)
Tensor(::Val{Constant}, ::Type{T}, x::Real; name=nothing) where {T<:Real} = push_constant!(fill(T(x)); name=name)
# Reject the value-less forms explicitly (the trampoline forwards `name`, so
# these must accept and ignore it):
Tensor(::Val{Constant}; name=nothing) =
    error("a Constant needs a value; provide an array or a float scalar, e.g. Tensor(Const, [1f0,2f0])")
Tensor(::Val{Constant}, dims::Integer...; name=nothing) =
    error("a Constant needs a value, not a shape ($(dims)); use Tensor(Const, array) or Tensor(Const, scalar)")
Tensor(::Val{Constant}, ::Type{<:Real}, dims::Integer...; name=nothing) =
    error("a Constant needs a value, not a shape; use Tensor(Const, array) or Tensor(Const, scalar)")

# --- bare data (no role word) => Constant ---------------------------
Tensor(data::AbstractArray{<:Real}; name=nothing) = push_constant!(_materialize(data); name=name)
Tensor(x::Real; name=nothing) = push_constant!(fill(x); name=name)

# --- lookup by (possibly scoped) name -------------------------------
Tensor(path::AbstractString...) = lookup(Symbol.(path))
