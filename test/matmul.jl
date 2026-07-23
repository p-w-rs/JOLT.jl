# =====================================================================
# ops/matmul.jl — `*` as 2-D matmul, symbolic contraction, and its vjp.
# =====================================================================

@testset "matmul (*)" begin

    @testset "forward: shape, IR, role" begin
        new_session!()
        A = Tensor(3, 4)
        B = Tensor(4, 5)
        C = A * B
        @test size(C) == (3, 5)
        @test roleof(C) == Result
        @test mlir_ranked(C) == (Float32, (3, 5))          # dot_general emitted the right type
        @test length(session().tape) == 1
    end

    @testset "shape rejection" begin
        new_session!()
        @test_throws ErrorException Tensor(3, 4) * Tensor(3, 5)     # inner dims 4 vs 3
        @test_throws ErrorException Tensor(2, 3, 4) * Tensor(4, 5)  # not 2-D
        @test_throws ErrorException Tensor(3, 4) * Tensor(4)        # not 2-D
        @test_throws ErrorException Tensor(Float32) * Tensor(Float32)  # scalars: not 2-D
        @test size(Tensor(0, 3) * Tensor(3, 4)) == (0, 4)           # 0-sized dim is legal
    end

    @testset "symbolic contraction" begin
        new_session!()
        A = Tensor(:M, :K)
        B = Tensor(:K, :N)
        C = A * B
        @test size(C) == (todim(:M), todim(:N))            # inner :K proved equal, cancels
        @test mlir_ranked(C) == (Float32, (:dyn, :dyn))

        @test_throws ErrorException Tensor(:M, :K) * Tensor(:J, :N) # K vs J not provable
        same_dim!(:K, :J)                                          # ...until declared
        @test size(Tensor(:M, :K) * Tensor(:J, :N)) == (todim(:M), todim(:N))
    end

    # Ā = ȳ·Bᵀ  (m,n)·(n,k) -> (m,k);  B̄ = Aᵀ·ȳ  (k,m)·(m,n) -> (k,n)
    @testset "vjp = ȳ·Bᵀ, Aᵀ·ȳ" begin
        s = new_session!()
        A = Tensor(3, 4)
        B = Tensor(4, 5)
        C = A * B
        ȳ = Tensor(3, 5)
        cts = JOLT.vjp(JOLT.MatMulOp(), ȳ, [A, B], C)
        @test size(cts[1]) == (3, 4)                       # Ā matches A
        @test size(cts[2]) == (4, 5)                       # B̄ matches B
        # symbolic case keeps symbols
        new_session!()
        As = Tensor(:M, :K); Bs = Tensor(:K, :N); Cs = As * Bs
        cs = JOLT.vjp(JOLT.MatMulOp(), Tensor(:M, :N), [As, Bs], Cs)
        @test size(cs[1]) == (todim(:M), todim(:K))
        @test size(cs[2]) == (todim(:K), todim(:N))
    end

    @testset "batched (rank ≥ 3)" begin
        new_session!()
        A = Tensor(2, 3, 4)
        B = Tensor(2, 4, 5)
        C = A * B                                          # (2,3,4)·(2,4,5) -> (2,3,5)
        @test size(C) == (2, 3, 5)
        @test mlir_ranked(C) == (Float32, (2, 3, 5))
        # deeper batch
        @test size(Tensor(2, 6, 3, 4) * Tensor(2, 6, 4, 5)) == (2, 6, 3, 5)
        # symbolic batch + contraction
        S = Tensor(:B, :M, :K) * Tensor(:B, :K, :N)
        @test size(S) == (todim(:B), todim(:M), todim(:N))
        @test mlir_ranked(S) == (Float32, (:dyn, :dyn, :dyn))

        # rejections
        @test_throws ErrorException Tensor(2, 3, 4) * Tensor(3, 4, 5)   # batch dim 2 vs 3
        @test_throws ErrorException Tensor(2, 3, 4) * Tensor(2, 5, 6)   # inner 4 vs 5
        @test_throws ErrorException Tensor(2, 3, 4) * Tensor(4, 5)      # rank mismatch 3 vs 2

        # vjp: Ā = ȳ·Bᵀ (batch-transpose), B̄ = Aᵀ·ȳ
        s = session()
        ȳ = Tensor(2, 3, 5)
        cts = JOLT.vjp(JOLT.MatMulOp(), ȳ, [A, B], C)
        @test size(cts[1]) == (2, 3, 4)                    # Ā matches A
        @test size(cts[2]) == (2, 4, 5)                    # B̄ matches B
    end

end
