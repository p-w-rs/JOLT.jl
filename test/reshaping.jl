# =====================================================================
# ops/reshaping.jl — reshape and transpose (permutedims / adjoint).
# =====================================================================

@testset "reshaping" begin

    @testset "reshape: forward, IR, count check" begin
        new_session!()
        x = Arg(2, 6)
        y = reshape(x, 3, 4)
        @test size(y) == (3, 4)
        @test y isa Result
        @test mlir_ranked(y) == (Float32, (3, 4))
        @test_throws ErrorException reshape(x, 3, 3)       # 12 elems ≠ 9
        @test size(reshape(x, 12)) == (12,)                # flatten
        @test size(reshape(Arg(Float32), 1)) == (1,)    # scalar -> (1,)
        # symbolic reshape must be PROVABLY count-preserving
        @test size(reshape(Arg(:B, 6), :B, 2, 3)) == (todim(:B), 2, 3)  # 6B == 6B
        @test_throws ErrorException reshape(Arg(:B, 4), :B)             # 4B ≠ B, rejected
        # negative target dims are rejected at build time (would mint invalid MLIR)
        @test_throws ErrorException reshape(Arg(6), -2, -3)            # negative dims
        @test_throws ErrorException reshape(Arg(12), -1, -12)         # -1 is not inference
    end

    @testset "reshape vjp restores the input shape" begin
        s = new_session!()
        x = Arg(2, 6)
        y = reshape(x, 3, 4)
        ȳ = Arg(3, 4)
        cts = JOLT.vjp(JOLT.ReshapeOp((3, 4)), ȳ, [x], y)
        @test size(cts[1]) == (2, 6)                       # reshaped back to x's shape
    end

    @testset "permutedims / adjoint" begin
        new_session!()
        x = Arg(2, 3)
        @test size(permutedims(x)) == (3, 2)               # default: reverse dims
        @test size(adjoint(x)) == (3, 2)                   # real: adjoint ≡ transpose
        @test size(permutedims(Arg(2, 3, 4), (3, 1, 2))) == (4, 2, 3)
        @test mlir_ranked(permutedims(x)) == (Float32, (3, 2))
        @test size(permutedims(Arg(Float32))) == ()     # scalar: empty permutation
        # a malformed permutation is a clean JOLT error, not a BoundsError / MLIR failure
        @test_throws ErrorException permutedims(Arg(2, 3), (1, 2, 3))   # wrong length
        @test_throws ErrorException permutedims(Arg(2, 3), (1, 1))      # not a permutation
    end

    @testset "transpose vjp = inverse permutation" begin
        new_session!()
        x = Arg(2, 3, 4)
        y = permutedims(x, (3, 1, 2))                      # -> (4, 2, 3)
        ȳ = Arg(4, 2, 3)
        cts = JOLT.vjp(JOLT.TransposeOp([3, 1, 2]), ȳ, [x], y)
        @test size(cts[1]) == (2, 3, 4)                    # back to x's shape
    end

    @testset "broadcasting (.+ .- .* ./ ⊙)" begin
        new_session!()
        z = Arg(3, 1) .+ Arg(1, 4)
        @test z isa AbstractTensor && size(z) == (3, 4)    # intercepted (not a Broadcasted), stretched
        @test size(Arg(2, 3) .- Arg(3)) == (2, 3)          # left-pad align
        @test size(Arg(2, 3) .* Arg(2, 3)) == (2, 3)       # same shape
        @test size(Arg(2, 3) ./ Arg(2, 3)) == (2, 3)       # ./ broadcasts (DivOp under the hood)
        @test size(Arg(32, 10) .+ Arg(10)) == (32, 10)     # (32,10).+(10,): bias added to each row (TF/JAX)
        @test size(Arg(32, 10) ./ Arg(10)) == (32, 10)
        @test size(Arg(3, 1) ⊙ Arg(1, 4)) == (3, 4)        # ⊙ ≡ .* (Hadamard, broadcasting)
        @test size(2f0 .* Arg(2, 3)) == (2, 3)             # scalar · tensor (dotted)
        @test size(2f0 * Arg(2, 3)) == (2, 3)              # ...and bare scalar * / /
        @test size(Arg(2, 3) / 2f0) == (2, 3)
        @test size(Arg(2, 3) .+ 1f0) == (2, 3)
        @test size(Arg(:B, 1) .+ Arg(:B, 4)) == (todim(:B), 4)  # dynamic dim CARRIED through
        @test_throws ErrorException Arg(3, 2) .+ Arg(4)         # 2 vs 4 not broadcastable
        # dynamic NEW axis (a bias over a :B batch) now builds — it lowers via
        # dynamic_broadcast_in_dim (runtime size read from the sibling with :B).
        @test size(Arg(:B, 4) .+ Arg(4)) == (todim(:B), 4)
        # still-unsupported dotted op (.^) / Array operand throws (curated messages deferred)
        @test_throws Exception Arg(2, 3) .^ Arg(2, 3)
        @test_throws Exception Arg(2, 3) .+ [1f0, 2f0, 3f0]
    end

    @testset "broadcast_to: forward + vjp reduces the stretched axes" begin
        new_session!()
        a = Arg(3, 1)
        bt = broadcast_to(a, (3, 4), [1, 2])               # stretch axis 2: 1 -> 4
        @test size(bt) == (3, 4)
        @test mlir_ranked(bt) == (Float32, (3, 4))
        ȳ = Arg(3, 4)
        cts = JOLT.vjp(JOLT.BroadcastToOp((3, 4), [1, 2]), ȳ, [a], bt)
        @test size(cts[1]) == (3, 1)                       # summed over the stretched axis, reshaped back
        # replicate into a NEW leading axis (as reduce_sum's vjp does)
        r = broadcast_to(Arg(4), (2, 4), [2])
        @test size(r) == (2, 4)
    end

    @testset "concatenate / slice / pad / axis_size" begin
        new_session!()
        a = Arg(2); b = Arg(3)
        c = concatenate(a, b)
        @test c isa Result && size(c) == (5,)
        @test size(concatenate(Arg(2, 2), Arg(2, 2); dim=1)) == (4, 2)
        @test size(concatenate(Arg(2, 2), Arg(2, 2); dim=2)) == (2, 4)
        cts = JOLT.vjp(JOLT.ConcatOp(1), Arg(5), [a, b], c)        # concat vjp = slice each slab
        @test size(cts[1]) == (2,) && size(cts[2]) == (3,)

        x = Arg(5)
        s = slice(x, 2:4)
        @test size(s) == (3,)
        @test size(JOLT.vjp(JOLT.SliceOp([2], [4]), Arg(3), [x], s)[1]) == (5,)   # vjp = pad
        @test_throws ErrorException slice(Arg(5), 3:9)                            # out of bounds

        p = pad(Arg(3); low=[2], high=[1])
        @test size(p) == (6,)
        @test size(JOLT.vjp(JOLT.PadOp([2], [1], 0.0), Arg(6), [Arg(3)], p)[1]) == (3,)   # vjp = slice

        @test axis_size(Arg(4, 3), 1) isa Constant                # static folds to a Constant
        @test axis_size(Arg(:B, 3), 1) isa Result && size(axis_size(Arg(:B, 3), 1)) == ()   # symbolic: runtime scalar
    end

end
