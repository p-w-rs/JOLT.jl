# =====================================================================
# Initializers
#
# An *initializer* is any FUNCTION with the signature
#
#       init(rng::AbstractRNG, T::Type, dims::Integer...) -> Array{T}
#
#   rng   — a random source. `compile` accepts a raw seed too, but normalizes it
#           to an AbstractRNG BEFORE calling init, so init always sees a concrete
#           RNG. Deterministic initializers ignore it.
#   T     — the element type to produce (e.g. Float32).
#   dims  — the concrete shape.
#
# A Variable carries one of these and JOLT calls it at COMPILE time to realize
# the Variable's value (nothing is allocated at graph-build time). The signature
# is checked when the Variable is built (`push_var!` uses `hasmethod`), so a
# mistyped init — wrong arity, or a function that doesn't take the T argument
# (e.g. `Base.ones`) — is rejected early rather than deferred to compile.
#
# Each name below is a plain function that BUILDS and returns such a closure, so
# you configure it once (gain, mean/std, fill value, …) and hand the result to a
# Variable:
#
#       Tensor(Var, 240, 10; init = GlorotUniform())
#       Tensor(Var, 10;      init = Fill(0.1))
#       Tensor(Var, 8, 8;    init = RandN(0, 0.02))
#
# If none of these fit, just pass your own closure with the same signature —
# there is nothing special about the ones defined here:
#
#       Tensor(Var, 8, 8; init = (rng, T, dims...) -> randn(rng, T, dims) ./ 10)
# =====================================================================

# --- deterministic ---------------------------------------------------
Zeros() = (rng::AbstractRNG, T::Type, dims::Integer...) -> zeros(T, dims...)
Ones()  = (rng::AbstractRNG, T::Type, dims::Integer...) -> ones(T, dims...)

"""Constant fill with `value` (cast to the requested element type)."""
Fill(value::Real) = (rng::AbstractRNG, T::Type, dims::Integer...) -> fill(convert(T, value), dims...)

# --- random ----------------------------------------------------------
# A rank-0 draw is Array{T,0}, but an affine broadcast (a .* x .+ b) collapses a
# 0-dim array back to a scalar — `_asarray` re-wraps it so every initializer
# honors the `-> Array{T}` contract for scalar Variables too.
_asarray(x::AbstractArray) = x
_asarray(x) = fill(x)

"""Normal draws with the given `mean` and standard deviation `std`."""
RandN(mean::Real = 0, std::Real = 1) =
    (rng::AbstractRNG, T::Type, dims::Integer...) ->
        _asarray(convert(T, std) .* randn(rng, T, dims) .+ convert(T, mean))

"""Uniform draws on the half-open interval `[lo, hi)`."""
Rand(lo::Real = 0, hi::Real = 1) =
    (rng::AbstractRNG, T::Type, dims::Integer...) ->
        _asarray(convert(T, lo) .+ (convert(T, hi) - convert(T, lo)) .* rand(rng, T, dims))

# fan-in / fan-out for the Glorot/He family: everything but the last dim feeds
# in, the last dim is the output width. (Scalars/vectors degrade gracefully.)
_fan(dims::Tuple) =
    (isempty(dims) ? 1 : prod(dims[1:end-1]),
     isempty(dims) ? 1 : dims[end])

"""Glorot/Xavier uniform: U(-limit, limit), limit = gain·√(6 / (fan_in + fan_out))."""
function GlorotUniform(gain::Real = 1)
    return (rng::AbstractRNG, T::Type, dims::Integer...) -> begin
        fan_in, fan_out = _fan(dims)
        limit = convert(T, gain) * sqrt(convert(T, 6) / convert(T, fan_in + fan_out))
        return _asarray((rand(rng, T, dims) .* (2 * limit)) .- limit)
    end
end

"""Glorot/Xavier normal: N(0, std²), std = gain·√(2 / (fan_in + fan_out))."""
function GlorotNormal(gain::Real = 1)
    return (rng::AbstractRNG, T::Type, dims::Integer...) -> begin
        fan_in, fan_out = _fan(dims)
        std = convert(T, gain) * sqrt(convert(T, 2) / convert(T, fan_in + fan_out))
        return _asarray(std .* randn(rng, T, dims))
    end
end
