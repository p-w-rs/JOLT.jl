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

        @testset "export_stablehlo (no backend)" begin
            new_session!()
            p = Tensor(3; name="p"); q = Tensor(Var, 3; init=Ones())
            txt = export_stablehlo([p], JOLT.Tensor[p + q])
            @test occursin("func.func @main", txt) && occursin("stablehlo.add", txt)
        end
    end
end
