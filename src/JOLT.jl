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
# tensor.jl defines Tensor, ops.jl defines Op/OpNode, and session.jl's Session
# holds all of them. Method *bodies* may still forward-reference functions
# defined later (resolved at call time) — which is how tensor.jl's constructors
# call `push_arg!`, ops.jl's `apply` calls `push_op!`, and dims.jl's
# convenience queries call `current_facts()` before session.jl is loaded.
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

include("ops.jl")       # Op/OpNode, apply, and the op surface
export mul

include("session.jl")   # Session, globals, facts/tape, name registry, IR builders
export Session, session, session!, new_session!, reset_session!, with_session
export current_facts
export default_dtype, default_dtype!
export namespace, pushnamespace!, popnamespace!, clearnamespace!

end # module JOLT
