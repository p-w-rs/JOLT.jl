# =====================================================================
# ops/reshaping.jl — reshape and transpose (permutedims / adjoint).
# =====================================================================

@testset "reshaping" begin

    @testset "reshape: forward, IR, count check" begin
        new_session!()
        x = Tensor(2, 6)
        y = reshape(x, 3, 4)
        @test size(y) == (3, 4)
        @test roleof(y) == Result
        @test mlir_ranked(y) == (Float32, (3, 4))
        @test_throws ErrorException reshape(x, 3, 3)       # 12 elems ≠ 9
        @test size(reshape(x, 12)) == (12,)                # flatten
        @test size(reshape(Tensor(Float32), 1)) == (1,)    # scalar -> (1,)
        # symbolic reshape must be PROVABLY count-preserving
        @test size(reshape(Tensor(:B, 6), :B, 2, 3)) == (todim(:B), 2, 3)  # 6B == 6B
        @test_throws ErrorException reshape(Tensor(:B, 4), :B)             # 4B ≠ B, rejected
        # negative target dims are rejected at build time (would mint invalid MLIR)
        @test_throws ErrorException reshape(Tensor(6), -2, -3)            # negative dims
        @test_throws ErrorException reshape(Tensor(12), -1, -12)         # -1 is not inference
    end

    @testset "reshape vjp restores the input shape" begin
        s = new_session!()
        x = Tensor(2, 6)
        y = reshape(x, 3, 4)
        ȳ = Tensor(3, 4)
        cts = JOLT.vjp(JOLT.ReshapeOp((3, 4)), ȳ, [x], y)
        @test size(cts[1]) == (2, 6)                       # reshaped back to x's shape
    end

    @testset "permutedims / adjoint" begin
        new_session!()
        x = Tensor(2, 3)
        @test size(permutedims(x)) == (3, 2)               # default: reverse dims
        @test size(adjoint(x)) == (3, 2)                   # real: adjoint ≡ transpose
        @test size(permutedims(Tensor(2, 3, 4), (3, 1, 2))) == (4, 2, 3)
        @test mlir_ranked(permutedims(x)) == (Float32, (3, 2))
        @test size(permutedims(Tensor(Float32))) == ()     # scalar: empty permutation
        # a malformed permutation is a clean JOLT error, not a BoundsError / MLIR failure
        @test_throws ErrorException permutedims(Tensor(2, 3), (1, 2, 3))   # wrong length
        @test_throws ErrorException permutedims(Tensor(2, 3), (1, 1))      # not a permutation
    end

    @testset "transpose vjp = inverse permutation" begin
        new_session!()
        x = Tensor(2, 3, 4)
        y = permutedims(x, (3, 1, 2))                      # -> (4, 2, 3)
        ȳ = Tensor(4, 2, 3)
        cts = JOLT.vjp(JOLT.TransposeOp([3, 1, 2]), ȳ, [x], y)
        @test size(cts[1]) == (2, 3, 4)                    # back to x's shape
    end

end
