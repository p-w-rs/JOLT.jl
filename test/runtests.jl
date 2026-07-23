using Test
using JOLT
import Optimisers   # test-only: verify a real optimizer step keeps `vars` zero-copy (numeric.jl)

# Tests are split by source module (tensor.jl, session.jl); shared introspection
# helpers live in util.jl. Every testset starts with a fresh session
# (new_session!), which isolates the name registry / input order and keeps the
# MLIR context stack balanced regardless of the order testsets run in.
include("util.jl")

@testset "JOLT" begin
    include("dims.jl")
    include("tensor.jl")
    include("session.jl")
    include("ops.jl")         # Op machinery + basic elementwise (add/sub/mul/neg)
    include("reduce.jl")      # reduce_sum, sum
    include("reshaping.jl")   # reshape, transpose, broadcasting (.+ .- .* ./ ⊙)
    include("matmul.jl")      # matmul / ⊡
    include("einsum.jl")      # einsum (general contraction), dot / ⋅
    include("gradients.jl")   # ∇, second order, stop_gradient, grad_reversal, broadcasting grads
    include("compile.jl")     # end-to-end IREE compile+run (skips if IREE isn't built)
    include("numeric.jl")     # end-to-end NUMERICAL check: every op + 1st/2nd-order grads vs references
end
