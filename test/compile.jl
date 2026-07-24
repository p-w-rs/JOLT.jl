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
            x = Arg(4; name="x")
            w = Var(Ones(), 4)
            b = Var(Fill(0.5f0), 4)
            y = x + w; z = mul(y, b); loss = sum(z)
            fn, vars = compile(0, [x], JOLT.AbstractTensor[y, z, loss])
            yv, zv, lv = fn(vars, Float32[1, 2, 3, 4])
            @test yv == Float32[2, 3, 4, 5]
            @test zv == Float32[1, 1.5, 2, 2.5]
            @test lv ≈ 7f0
        end

        @testset "gradient VALUES: d(a·b+a) = (b+1, a)" begin
            new_session!()
            a = Var(Float32, Fill(3f0))
            b = Var(Float32, Fill(2f0))
            L = mul(a, b) + a
            g = ∇(L; wrt=[a, b])
            fn, vars = compile(0, JOLT.AbstractTensor[], JOLT.AbstractTensor[L, g[1], g[2]])
            Lv, ga, gb = fn(vars)
            @test Lv ≈ 9f0 && ga ≈ 3f0 && gb ≈ 3f0
        end

        @testset "multi-dim: matmul forward + gradient vs reference" begin
            new_session!()
            X = Arg(2, 3; name="X")
            W = Var(Ones(), 3, 4)
            Y = matmul(X, W); loss = sum(Y); gW = ∇(loss; wrt=W)
            fn, vars = compile(0, [X], JOLT.AbstractTensor[Y, gW])
            Xv = Float32[1 2 3; 4 5 6]
            Yv, gWv = fn(vars, Xv)
            @test Yv ≈ Xv * ones(Float32, 3, 4)
            @test gWv ≈ permutedims(Xv) * ones(Float32, 2, 4)
        end

        @testset "vars is a ComponentArray of :variables" begin
            new_session!()
            namespace("params") do
                Var(Ones(), 2, 2; name="W")
            end
            _, vars = compile(0, JOLT.AbstractTensor[], JOLT.AbstractTensor[Const(3.0f0)])
            @test vars isa JOLT.ComponentArray
            @test vars.params.W == ones(Float32, 2, 2)
        end

        @testset "zero-copy weight READ is LIVE (arena as call input)" begin
            # With no assign!, `vars` rides as a READ-ONLY call input (arg 0),
            # imported zero-copy from the ComponentArray's page-aligned arena, so an
            # in-place mutation between calls is seen by the next call with no copy.
            new_session!()
            x = Arg(2; name="x")
            w = Var(Ones(), 2; name="w")
            fn, vars = compile(0, [x], JOLT.AbstractTensor[x + w])
            @test UInt(pointer(JOLT.getdata(vars))) % 64 == 0        # import alignment
            @test fn(vars, Float32[10, 20]) == Float32[11, 21]       # w = [1, 1]
            @test fn.exe.zerocopy == 1                               # CPU heap imports zero-copy
            JOLT.getdata(vars) .= Float32[100, 200]                  # mutate the SHARED buffer in place
            @test fn(vars, Float32[10, 20]) == Float32[110, 220]     # IREE sees it → live
        end

        @testset "state write-back (assign!): read-old, advance in place" begin
            # A variable read for an output AND assign!ed: the output sees the OLD
            # value (read-old/write-new), the new value comes back as an extra
            # output, and `fn` copies it into the SAME arena slot (pointer unchanged,
            # only the changed variable copied).
            new_session!()
            x = Arg(3; name="x")
            W = Var((rng, T, d...) -> T[1, 1, 1], 3; name="W")
            y = W .* x; assign!(W, W .+ x)                           # y uses OLD W; W advances
            fn, vars = compile(0, [x], JOLT.AbstractTensor[sum(y)])
            p0 = pointer(JOLT.getdata(vars))
            @test fn(vars, Float32[1, 2, 3]) ≈ 6f0                   # sum(oldW .* x) = 6 (NOT 20)
            @test collect(vars.default.W) == Float32[2, 3, 4]        # W advanced
            @test pointer(JOLT.getdata(vars)) == p0                  # same arena — no reallocation
            @test fn.exe.zerocopy == 1                               # reads are zero-copy (CPU heap)
            @test fn(vars, Float32[1, 2, 3]) ≈ 20f0                  # now W = [2,3,4]
            @test collect(vars.default.W) == Float32[3, 5, 7]
        end

        @testset "snapshot / pure and the congruent-vars guard" begin
            new_session!()
            x = Arg(3; name="x")
            W = Var((rng, T, d...) -> T[1, 1, 1], 3; name="W")
            y = W .* x; assign!(W, W .+ x)
            fn, vars = compile(0, [x], JOLT.AbstractTensor[sum(y)])

            snap = snapshot(vars)                                    # fresh page-aligned arena
            @test UInt(pointer(JOLT.getdata(snap))) % 64 == 0
            @test collect(snap.default.W) == Float32[1, 1, 1]
            fn(vars, Float32[1, 2, 3])                               # advances vars, not snap
            @test collect(vars.default.W) == Float32[2, 3, 4]
            @test collect(snap.default.W) == Float32[1, 1, 1]        # snapshot untouched

            y2, nv = pure(fn, vars, Float32[1, 2, 3])               # vars=[2,3,4]: y=[2,6,12]→20
            @test y2 ≈ 20f0
            @test collect(vars.default.W) == Float32[2, 3, 4]        # pure left the input unchanged
            @test collect(nv.default.W)  == Float32[3, 5, 7]         # advanced snapshot returned

            @test fn(snap, Float32[1, 2, 3]) ≈ 6f0                   # a congruent (snapshot) vars is accepted

            # a layout-mismatched vars is rejected
            new_session!()
            x2 = Arg(2; name="x"); w2 = Var(Ones(), 2; name="w")
            fn2, _ = compile(0, [x2], JOLT.AbstractTensor[x2 + w2])
            @test_throws Exception fn2(vars, Float32[1, 2])          # wrong length/axes
        end

        @testset "Metal backend: zero-copy weights live on the GPU" begin
            # Needs the Metal HAL driver (built here) + a Metal device. Compiles
            # with MSL embedded (runtime-compiled via Metal.framework), so no Xcode
            # `metal`/`metallib` CLI is required. Skips where neither is available.
            metal_ok = try
                new_session!()
                xp = Arg(1; name="x"); wp = Var(Ones(), 1; name="w")
                fp, vp = compile(0, [xp], JOLT.AbstractTensor[xp + wp]; backend=JOLT.IREE_METAL)
                fp(vp, Float32[1]); true
            catch; false end
            if !metal_ok
                @info "Metal tests skipped — no Metal device / runtime available"
            else
                new_session!()
                x = Arg(2; name="x"); w = Var(Ones(), 2; name="w")
                fn, vars = compile(0, [x], JOLT.AbstractTensor[x + w]; backend=JOLT.IREE_METAL)
                @test fn(vars, Float32[10, 20]) == Float32[11, 21]
                JOLT.getdata(vars) .= Float32[100, 200]          # mutate shared arena in place
                @test fn(vars, Float32[10, 20]) == Float32[110, 220]  # GPU sees it → live, zero-copy read

                # state write-back on the GPU: reads are zero-copy from the shared
                # arena (unified memory), the assigned variable advances read-old,
                # and the arena is not reallocated.
                new_session!()
                xx = Arg(3; name="x")
                W = Var((rng, T, d...) -> T[1, 1, 1], 3; name="W")
                yy = W .* xx; assign!(W, W .+ xx)
                fw, vw = compile(0, [xx], JOLT.AbstractTensor[sum(yy)]; backend=JOLT.IREE_METAL)
                p0 = pointer(JOLT.getdata(vw))
                @test fw(vw, Float32[1, 2, 3]) ≈ 6f0             # read-old on the GPU
                @test collect(vw.default.W) == Float32[2, 3, 4]  # advanced
                @test pointer(JOLT.getdata(vw)) == p0            # same arena — no reallocation
            end
        end

        @testset "export_stablehlo (no backend)" begin
            new_session!()
            p = Arg(3; name="p"); q = Var(Ones(), 3)
            txt = export_stablehlo([p], JOLT.AbstractTensor[p + q])
            @test occursin("func.func @main", txt) && occursin("stablehlo.add", txt)
        end
    end
end
