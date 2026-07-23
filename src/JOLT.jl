"""
    𝐉𝐎𝐋𝐓

    𝐉ulia StableHL𝐎 𝐋earning 𝐓ensors
"""
module JOLT

# JOLT builds StableHLO directly (no bespoke IR) and hands it to a compiler backend (like IREE).
# We reuse Reactant *only* for its MLIR bindings + StableHLO dialect builders — never its tracing / XLA / PJRT.
# These four aliases are the single point of coupling to Reactant.
using Reactant: Reactant
const MLIR  = Reactant.MLIR
const IR    = MLIR.IR
const shlo  = MLIR.Dialects.stablehlo
const funcd = MLIR.Dialects.func

using Random   # rng-driven initializers (randn/rand) in initializers.jl

# Include order matters for TYPES (each file's structs must exist before a
# later file mentions them in a field or signature): dims.jl defines Dim/Facts,
# tensor.jl defines Tensor, ops/core.jl defines Op/OpNode (a field of Session),
# and session.jl's Session holds all of them. The concrete ops come AFTER
# session.jl because their `lower` bodies call `push_op!`; ops/gradients.jl is
# last (it walks the tape). Method *bodies* may still forward-reference
# functions defined later (resolved at call time) — which is how ops/core.jl's
# `apply` reaches `push_op!` and dims.jl's queries reach `current_facts()`.
include("dims.jl")      # symbolic dims: polynomial algebra, Facts, analyzers
export Dim, Poly, todim, realize, canon, dims_equal
export same_dim!, pin!, divisible!, bound!, check_facts
export provably_divisible, provably_ge

include("initializers.jl")  # closure-based Variable initializers
export Zeros, Ones, Fill, RandN, Rand, GlorotUniform, GlorotNormal

include("tensor.jl")    # roles, Tensor, traits, public constructors
export AbstractTensor, Tensor, roleof
export getvalue
export TensorRole, Argument, Arg, Variable, Var, Constant, Const, Result, Res

include("ops/core.jl")  # Op/OpNode/apply machinery (OpNode is a Session field)

include("session.jl")   # Session, globals, facts/tape, name registry, IR builders
export Session, session, session!, new_session!, reset_session!, with_session
export current_facts
export default_dtype, default_dtype!
export namespace, pushnamespace!, popnamespace!, clearnamespace!

include("ops/basic.jl")       # add, subtract, multiply (mul), negate
include("ops/reshaping.jl")   # reshape, transpose (permutedims), adjoint
include("ops/matmul.jl")      # * — 2-D matmul
include("ops/gradients.jl")   # ∇/gradient, stop_gradient, grad_reversal
export mul, ∇, gradient, stop_gradient, grad_reversal

end # module JOLT
