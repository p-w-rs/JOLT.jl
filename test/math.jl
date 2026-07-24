# =====================================================================
# ops/math.jl — unary activations, select, compare, dotted comparisons.
# (construction/shape/vjp here; numeric behaviour lives in numeric.jl)
# =====================================================================

@testset "math (activations, select, compare)" begin

    @testset "unary activations: shape, role, vjp reads the output" begin
        new_session!()
        x = Arg(2, 3)
        for y in (sqrt(x), rsqrt(x), tanh(x), sigmoid(x), relu(x))
            @test y isa Result && size(y) == (2, 3)
        end
        @test mlir_ranked(tanh(x)) == (Float32, (2, 3))
        ȳ = Arg(2, 3)
        @test size(JOLT.vjp(JOLT.SqrtOp(), ȳ, [x], sqrt(x))[1]) == (2, 3)
        @test size(JOLT.vjp(JOLT.TanhOp(), ȳ, [x], tanh(x))[1]) == (2, 3)
    end

    @testset "compare -> i1, non-differentiable" begin
        new_session!()
        x = Arg(3); y = Arg(3)
        p = x .> y
        @test p isa Result && size(p) == (3,) && eltype(p) == Bool
        @test all(t -> t isa Result && eltype(t) == Bool, (x .== y, x .!= y, x .<= 0f0, x .>= 1f0, x .< y))
        @test JOLT.vjp(JOLT.CompareOp("GT"), Arg(3), [x, y], p) == (nothing, nothing)
    end

    @testset "select: shapes, scalar predicate, mixed eltype, vjp routing" begin
        new_session!()
        x = Arg(2, 3); z = Arg(2, 3)
        s = select(x .> z, x, z)
        @test s isa Result && size(s) == (2, 3)
        f = Arg()                                              # scalar predicate broadcasts
        @test size(select(f .!= 0f0, x, z)) == (2, 3)
        # apply's eltype check is relaxed for SelectOp (i1 pred + float branches)
        @test size(select(Arg(2) .> 0f0, Arg(Float32, 2), Arg(Float32, 2))) == (2,)
        @test_throws ErrorException select(Arg(2,3) .> Arg(2,3), Arg(2,3), Arg(3,2))   # branch mismatch
        ȳ = Arg(2, 3)
        cts = JOLT.vjp(JOLT.SelectOp(), ȳ, [x .> z, x, z], s)
        @test cts[1] === nothing && size(cts[2]) == (2, 3) && size(cts[3]) == (2, 3)
    end

end
