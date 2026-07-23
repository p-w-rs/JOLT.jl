# =====================================================================
# compile/ — end-to-end IREE integration tests.
#
# These need the IREE compiler + runtime shim built (via `] build JOLT`). When
# they're absent (fresh checkout / CI without the build) the whole set skips, so
# the rest of the suite still runs. When present, they verify the full stack —
# JOLT graph → StableHLO → iree-compile → runtime — including gradient VALUES.
# =====================================================================

@testset "compile (IREE)" begin
    have_iree = isfile(JOLT.IREEBuild.runtime_lib(JOLT._UUID)) &&
                isfile(JOLT.IREEBuild.compiler_bin(JOLT._UUID))
    if !have_iree
        @info "compile (IREE) tests skipped — runtime/compiler not built (`] build JOLT`)"
    else
        @testset "forward: elementwise + reduce, vars + args" begin
            new_session!()
            x = Tensor(4; name="x")
            w = Tensor(Var, 4; init=Ones())
            b = Tensor(Var, 4; init=Fill(0.5f0))
            y = x + w; z = mul(y, b); loss = sum(z)
            fn, vars = compile(0, [x], JOLT.Tensor[y, z, loss])
            yv, zv, lv = fn(vars, Float32[1, 2, 3, 4])
            @test yv == Float32[2, 3, 4, 5]
            @test zv == Float32[1, 1.5, 2, 2.5]
            @test lv ≈ 7f0
        end

        @testset "gradient VALUES: d(a·b+a) = (b+1, a)" begin
            new_session!()
            a = Tensor(Var, Float32; init=Fill(3f0))
            b = Tensor(Var, Float32; init=Fill(2f0))
            L = mul(a, b) + a
            g = ∇(L; wrt=[a, b])
            fn, vars = compile(0, JOLT.Tensor[], JOLT.Tensor[L, g[1], g[2]])
            Lv, ga, gb = fn(vars)
            @test Lv ≈ 9f0 && ga ≈ 3f0 && gb ≈ 3f0
        end

        @testset "multi-dim: matmul forward + gradient vs reference" begin
            new_session!()
            X = Tensor(2, 3; name="X")
            W = Tensor(Var, 3, 4; init=Ones())
            Y = X * W; loss = sum(Y); gW = ∇(loss; wrt=W)
            fn, vars = compile(0, [X], JOLT.Tensor[Y, gW])
            Xv = Float32[1 2 3; 4 5 6]
            Yv, gWv = fn(vars, Xv)
            @test Yv ≈ Xv * ones(Float32, 3, 4)
            @test gWv ≈ permutedims(Xv) * ones(Float32, 2, 4)
        end

        @testset "vars is a ComponentArray of :variables" begin
            new_session!()
            namespace("params") do
                Tensor(Var, 2, 2; init=Ones(), name="W")
            end
            _, vars = compile(0, JOLT.Tensor[], JOLT.Tensor[Tensor(3.0f0)])
            @test vars isa JOLT.ComponentArray
            @test vars.params.W == ones(Float32, 2, 2)
        end

        @testset "zero-copy weights are LIVE (io_parameters + barrier)" begin
            # `vars` is served to IREE by an io_parameters provider that ALIASES
            # the ComponentArray's page-aligned arena (not pushed as a call input),
            # so an in-place mutation is seen by the next call with no copy. This
            # also pins the optimization_barrier workaround: without it iree-compile
            # 3.12 segfaults folding a flow.tensor.slice of a #flow.parameter, so a
            # successful compile here means the workaround is in place.
            new_session!()
            x = Tensor(2; name="x")
            w = Tensor(Var, 2; init=Ones(), name="w")
            fn, vars = compile(0, [x], JOLT.Tensor[x + w])
            @test UInt(pointer(JOLT.getdata(vars))) % 64 == 0        # import alignment
            @test fn(vars, Float32[10, 20]) == Float32[11, 21]       # w = [1, 1]
            JOLT.getdata(vars) .= Float32[100, 200]                  # mutate the SHARED buffer in place
            @test fn(vars, Float32[10, 20]) == Float32[110, 220]     # IREE sees it → live
        end

        @testset "compiled fn rejects a foreign/reallocated vars" begin
            new_session!()
            x = Tensor(2; name="x"); w = Tensor(Var, 2; init=Ones(), name="w")
            fn, vars = compile(0, [x], JOLT.Tensor[x + w])
            fn(vars, Float32[1, 1])                                  # opens the session (binds the arena)
            @test_throws Exception fn(deepcopy(vars), Float32[1, 1]) # a different arena is not the shared one
        end

        @testset "Metal backend: zero-copy weights live on the GPU" begin
            # Needs the Metal HAL driver (built here) + a Metal device. Compiles
            # with MSL embedded (runtime-compiled via Metal.framework), so no Xcode
            # `metal`/`metallib` CLI is required. Skips where neither is available.
            metal_ok = try
                new_session!()
                xp = Tensor(1; name="x"); wp = Tensor(Var, 1; init=Ones(), name="w")
                fp, vp = compile(0, [xp], JOLT.Tensor[xp + wp]; backend=JOLT.IREE_METAL)
                fp(vp, Float32[1]); true
            catch; false end
            if !metal_ok
                @info "Metal tests skipped — no Metal device / runtime available"
            else
                new_session!()
                x = Tensor(2; name="x"); w = Tensor(Var, 2; init=Ones(), name="w")
                fn, vars = compile(0, [x], JOLT.Tensor[x + w]; backend=JOLT.IREE_METAL)
                @test fn(vars, Float32[10, 20]) == Float32[11, 21]
                JOLT.getdata(vars) .= Float32[100, 200]          # mutate shared arena in place
                @test fn(vars, Float32[10, 20]) == Float32[110, 220]  # GPU sees it → live, zero-copy
            end
        end

        @testset "export_stablehlo (no backend)" begin
            new_session!()
            p = Tensor(3; name="p"); q = Tensor(Var, 3; init=Ones())
            txt = export_stablehlo([p], JOLT.Tensor[p + q])
            @test occursin("func.func @main", txt) && occursin("stablehlo.add", txt)
        end
    end
end
