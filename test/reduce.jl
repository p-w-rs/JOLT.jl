# =====================================================================
# ops/reduce.jl — reduce_sum / sum (the reduction-region op) and its vjp.
# =====================================================================

@testset "reduce (sum / reduce_sum)" begin

    @testset "forward: shape, IR" begin
        new_session!()
        x = Tensor(2, 3, 4)
        @test size(sum(x)) == ()                       # full reduction -> rank-0 scalar
        @test mlir_ranked(sum(x)) == (Float32, ())
        @test size(reduce_sum(x, (2,)))   == (2, 4)     # drop axis 2
        @test size(reduce_sum(x, (1, 3))) == (3,)       # drop axes 1 and 3
        @test size(reduce_sum(x, (1, 2, 3))) == ()      # drop all
        @test roleof(sum(x)) == Result
        @test_throws ErrorException reduce_sum(x, (4,))  # axis out of range
        # symbolic dims survive
        @test size(sum(Tensor(:B, 4))) == ()
        @test size(reduce_sum(Tensor(:B, 4), (2,))) == (todim(:B),)
    end

    @testset "vjp replicates ȳ over the reduced axes" begin
        new_session!()
        x = Tensor(2, 3, 4)
        y = reduce_sum(x, (2,))                         # (2,4)
        ȳ = Tensor(2, 4)
        cts = JOLT.vjp(JOLT.ReduceSumOp([2]), ȳ, [x], y)
        @test size(cts[1]) == (2, 3, 4)                 # broadcast back over axis 2
        # full reduction: scalar cotangent broadcasts to the whole input
        z = sum(x)
        cz = JOLT.vjp(JOLT.ReduceSumOp([1, 2, 3]), Tensor(), [x], z)
        @test size(cz[1]) == (2, 3, 4)
    end

end
