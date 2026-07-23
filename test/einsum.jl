# =====================================================================
# ops/einsum.jl — general Einstein summation (built on reduce_sum / permutedims
# / reshape / matmul), and the `dot` / `⋅` inner product.
#
# Shape/graph/reject tests here; end-to-end NUMERICAL correctness (forward and
# gradients, incl. through IREE) lives in numeric.jl.
# =====================================================================

@testset "einsum" begin

    @testset "forward shapes: the canonical contractions" begin
        new_session!()
        @test size(einsum("ik,kj->ij", Arg(3, 4), Arg(4, 5))) == (3, 5)      # matmul
        @test size(einsum("bik,bkj->bij", Arg(2, 3, 4), Arg(2, 4, 5))) == (2, 3, 5)  # batched
        @test size(einsum("i,i->", Arg(5), Arg(5))) == ()                    # dot
        @test size(einsum("i,j->ij", Arg(3), Arg(4))) == (3, 4)              # outer
        @test size(einsum("ij->ji", Arg(3, 4))) == (4, 3)                    # transpose
        @test size(einsum("ij->i", Arg(3, 4))) == (3,)                       # row-sum
        @test size(einsum("ij->", Arg(3, 4))) == ()                          # full sum
        @test size(einsum("ij,jk,kl->il", Arg(2,3), Arg(3,4), Arg(4,5))) == (2, 5)   # 3-operand chain
        @test size(einsum("abcd,cd->ab", Arg(2,3,4,5), Arg(4,5))) == (2, 3)  # multi-index contract
    end

    @testset "results are Result tensors with the right IR type" begin
        new_session!()
        C = einsum("ik,kj->ij", Arg(3, 4), Arg(4, 5))
        @test C isa Result && mlir_ranked(C) == (Float32, (3, 5))
        # transpose-only einsum still yields a usable tensor
        @test einsum("ij->ji", Arg(3, 4)) isa AbstractTensor
    end

    @testset "symbolic dims flow through; shared index must be provably equal" begin
        new_session!()
        S = einsum("bik,bkj->bij", Arg(:B, :M, :K), Arg(:B, :K, :N))
        @test size(S) == (todim(:B), todim(:M), todim(:N))
        @test mlir_ranked(S) == (Float32, (:dyn, :dyn, :dyn))
        @test_throws ErrorException einsum("ik,kj->ij", Arg(:M, :K), Arg(:J, :N))  # K vs J unprovable
        same_dim!(:K, :J)
        @test size(einsum("ik,kj->ij", Arg(:M, :K), Arg(:J, :N))) == (todim(:M), todim(:N))
    end

    @testset "rejections: repeated index, bad output, ellipsis, arity, rank" begin
        new_session!()
        @test_throws ErrorException einsum("ii->i", Arg(3, 3))       # diagonal (repeat in operand)
        @test_throws ErrorException einsum("ii->", Arg(3, 3))        # trace (repeat in operand)
        @test_throws ErrorException einsum("i->ij", Arg(3))          # output index with no source
        @test_throws ErrorException einsum("ij->ii", Arg(3, 3))      # repeated output index
        @test_throws ErrorException einsum("...ij->ij", Arg(2, 3))   # ellipsis unsupported
        @test_throws ErrorException einsum("ij,jk->ik", Arg(3, 4))   # arity: 2 groups, 1 tensor
        @test_throws ErrorException einsum("ijk->ij", Arg(3, 4))     # rank: group names 3, tensor is 2
        @test_throws ErrorException einsum("ij", Arg(3, 4))          # no `->`
    end

    # einsum is decomposed into taped ops, so its gradient is ordinary graph —
    # here we just confirm it builds a cotangent of the right shape for each operand.
    @testset "vjp: shapes for matmul- and outer-form einsums" begin
        new_session!()
        X = Var(3, 4); W = Var(4, 5)
        L = sum(einsum("ik,kj->ij", X, W))
        g = ∇(L; wrt=[X, W])
        @test size(g[1]) == (3, 4) && size(g[2]) == (4, 5)

        a = Var(3); b = Var(4)
        Lo = sum(einsum("i,j->ij", a, b))
        go = ∇(Lo; wrt=[a, b])
        @test size(go[1]) == (3,) && size(go[2]) == (4,)
    end

    @testset "dot / ⋅ : the inner product" begin
        new_session!()
        @test size(dot(Arg(5), Arg(5))) == ()             # vectors -> scalar
        @test size(Arg(5) ⋅ Arg(5)) == ()                 # ⋅ is the same
        @test size(dot(Arg(3, 4), Arg(3, 4))) == ()        # Frobenius (full contraction)
        @test_throws ErrorException dot(Arg(3), Arg(4))    # rank ok but size mismatch (unprovable)
        @test_throws ErrorException dot(Arg(3), Arg(3, 4)) # unequal rank
        # gradient of a dot: ∂/∂a = b, ∂/∂b = a (shapes)
        a = Var(5); b = Var(5); g = ∇(dot(a, b); wrt=[a, b])
        @test size(g[1]) == (5,) && size(g[2]) == (5,)
    end

end
