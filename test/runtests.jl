using Test
using JOLT

# Tests are split by source module (tensor.jl, session.jl); shared introspection
# helpers live in util.jl. Every testset starts with a fresh session
# (new_session!), which isolates the name registry / input order and keeps the
# MLIR context stack balanced regardless of the order testsets run in.
include("util.jl")

@testset "JOLT" begin
    include("dims.jl")
    include("tensor.jl")
    include("session.jl")
    include("ops.jl")
end
