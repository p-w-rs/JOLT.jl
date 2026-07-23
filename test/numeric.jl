# =====================================================================
# numeric.jl — END-TO-END NUMERICAL verification.
#
# Every op (forward), every first-order gradient, and second-order (grad-of-grad)
# cases are built as a JOLT graph, COMPILED and RUN through IREE, and compared to
# an independently hand-computed Julia reference — on CPU, and on Metal when a GPU
# is available. This is the numeric counterpart to the op unit tests (which only
# check graph/shape/vjp construction, not backend execution). Skips whole when
# IREE isn't built (fresh checkout / CI without `] build JOLT`).
# =====================================================================

@testset "numeric end-to-end (IREE)" begin
    have_iree = isfile(JOLT.IREEBuild.runtime_lib(JOLT._UUID)) &&
                isfile(JOLT.IREEBuild.compiler_bin(JOLT._UUID))
    if !have_iree
        @info "numeric (IREE) tests skipped — runtime/compiler not built (`] build JOLT`)"
    else
        vinit(A)     = (rng, T, dims...) -> convert(Array{T}, A)   # Variable initialized to exactly A
        approx(a, b) = isapprox(collect(Float32.(a)), collect(Float32.(b)); rtol = 1f-4, atol = 1f-4)
        E  = JOLT.Tensor[]
        A2 = Float32[1 2 3; 4 5 6]
        B2 = Float32[6 5 4; 3 2 1]
        V3(x) = Tensor(Var, 3; init = vinit(Float32.(x)))

        # A case builds a FRESH graph and returns (inputs, outputs, argvals, refs);
        # `run1` compiles+runs it on `bk` and returns whether every output matches.
        function run1(bk, thunk)
            new_session!()
            inputs, outputs, argvals, refs = thunk()
            fn, vars = compile(0, inputs, outputs; backend = bk)
            got = fn(vars, argvals...); got = got isa Tuple ? collect(got) : [got]
            return length(got) == length(refs) && all(approx(g, r) for (g, r) in zip(got, refs))
        end

        cases = [
            # ---- forward (operands are Variables ⇒ exercises the zero-copy vars path) ----
            "fwd add"            => () -> (E, JOLT.Tensor[Tensor(Var,2,3;init=vinit(A2)) + Tensor(Var,2,3;init=vinit(B2))], (), [A2 .+ B2]),
            "fwd subtract"       => () -> (E, JOLT.Tensor[Tensor(Var,2,3;init=vinit(A2)) - Tensor(Var,2,3;init=vinit(B2))], (), [A2 .- B2]),
            "fwd mul"            => () -> (E, JOLT.Tensor[mul(Tensor(Var,2,3;init=vinit(A2)), Tensor(Var,2,3;init=vinit(B2)))], (), [A2 .* B2]),
            "fwd negate"         => () -> (E, JOLT.Tensor[-Tensor(Var,2,3;init=vinit(A2))], (), [-A2]),
            "fwd sum"            => () -> (E, JOLT.Tensor[sum(Tensor(Var,2,3;init=vinit(A2)))], (), [sum(A2)]),
            "fwd reduce_sum(2)"  => () -> (E, JOLT.Tensor[reduce_sum(Tensor(Var,2,3;init=vinit(A2)),(2,))], (), [dropdims(sum(A2;dims=2);dims=2)]),
            "fwd reshape(3,2)"   => () -> (E, JOLT.Tensor[reshape(Tensor(Var,2,3;init=vinit(A2)),3,2)], (), [reshape(A2,3,2)]),
            "fwd transpose"      => () -> (E, JOLT.Tensor[permutedims(Tensor(Var,2,3;init=vinit(A2)),(2,1))], (), [permutedims(A2)]),
            "fwd broadcast_to"   => () -> (E, JOLT.Tensor[broadcast_to(Tensor(Var,3,1;init=vinit(Float32[1;2;3;;])),(3,4),[1,2])], (), [repeat(Float32[1;2;3;;],1,4)]),
            "fwd .+ scalar"      => () -> (E, JOLT.Tensor[Tensor(Var,2,3;init=vinit(A2)) .+ 1f0], (), [A2 .+ 1f0]),
            "fwd .+ leftpad vec" => () -> (E, JOLT.Tensor[Tensor(Var,2,3;init=vinit(A2)) .+ Tensor(Var,3;init=vinit(Float32[10,20,30]))], (), [A2 .+ reshape(Float32[10,20,30],1,3)]),
            "fwd matmul 2D"      => function ()
                Wm = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
                X = Tensor(2,3; name="X"); W = Tensor(Var,3,4; init=vinit(Wm))
                (JOLT.Tensor[X], JOLT.Tensor[X * W], (A2,), [A2 * Wm])
            end,
            "fwd matmul batched" => function ()
                A3 = permutedims(cat(Float32[1 0 1; 0 1 0], 2Float32[1 0 1; 0 1 0]; dims=3), (3,1,2))       # (2,2,3)
                Wb = permutedims(cat(Float32[1 0 0 1; 0 1 0 1; 0 0 1 1], Float32[1 0 0 1; 0 1 0 1; 0 0 1 1]; dims=3), (3,1,2))  # (2,3,4)
                A = Tensor(Var,2,2,3; init=vinit(A3)); B = Tensor(Var,2,3,4; init=vinit(Wb))
                ref = similar(A3, 2,2,4); for i in 1:2; ref[i,:,:] = A3[i,:,:] * Wb[i,:,:]; end
                (E, JOLT.Tensor[A * B], (), [ref])
            end,
            # ---- first-order gradients ----
            "grad add"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(a+b);      JOLT.Tensor[L, ∇(L;wrt=a), ∇(L;wrt=b)]), (), [21f0, ones(Float32,3), ones(Float32,3)]),
            "grad sub"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(a-b);      JOLT.Tensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [ones(Float32,3), -ones(Float32,3)]),
            "grad mul"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(a,b)); JOLT.Tensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [Float32[4,5,6], Float32[1,2,3]]),
            "grad transpose"     => () -> (E, (a=Tensor(Var,2,3;init=vinit(A2)); t=permutedims(a,(2,1)); L=sum(mul(t,t)); JOLT.Tensor[∇(L;wrt=a)]), (), [2 .* A2]),
            "grad reshape"       => () -> (E, (a=Tensor(Var,2,3;init=vinit(A2)); r=reshape(a,3,2); L=sum(mul(r,r)); JOLT.Tensor[∇(L;wrt=a)]), (), [2 .* A2]),
            "grad matmul"        => function ()
                X = Tensor(2,3; name="X"); W = Tensor(Var,3,4; init=Ones()); L = sum(X*W)
                (JOLT.Tensor[X], JOLT.Tensor[∇(L;wrt=W)], (A2,), [permutedims(A2)*ones(Float32,2,4)])
            end,
            "grad stop_gradient" => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(stop_gradient(a),b)); JOLT.Tensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [zeros(Float32,3), Float32[1,2,3]]),
            "grad grad_reversal" => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(grad_reversal(a),b)); JOLT.Tensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [-Float32[4,5,6], Float32[1,2,3]]),
            "grad negate"        => () -> (E, (a=V3([1,2,3]); L=sum(-a); JOLT.Tensor[∇(L;wrt=a)]), (), [-ones(Float32,3)]),
            "grad reduce_sum"    => () -> (E, (a=Tensor(Var,2,3;init=vinit(A2)); r=reduce_sum(a,(2,)); L=sum(mul(r,r)); JOLT.Tensor[∇(L;wrt=a)]), (), [2 .* repeat(sum(A2;dims=2),1,3)]),
            "grad broadcast_to"  => () -> (E, (a=Tensor(Var,3,1;init=vinit(Float32[1;2;3;;])); L=sum(broadcast_to(a,(3,4),[1,2])); JOLT.Tensor[∇(L;wrt=a)]), (), [fill(4f0,3,1)]),
            "grad .+ broadcast"  => () -> (E, (a=Tensor(Var,2,3;init=vinit(A2)); b=Tensor(Var,3;init=vinit(Float32[10,20,30])); L=sum(a .+ b); JOLT.Tensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [ones(Float32,2,3), fill(2f0,3)]),
            "grad matmul batched" => function ()
                A3 = permutedims(cat(Float32[1 0 1; 0 1 0], 2Float32[1 0 1; 0 1 0]; dims=3), (3,1,2))
                Wb = permutedims(cat(Float32[1 0 0 1; 0 1 0 1; 0 0 1 1], Float32[1 0 0 1; 0 1 0 1; 0 0 1 1]; dims=3), (3,1,2))
                A = Tensor(Var,2,2,3; init=vinit(A3)); B = Tensor(Var,2,3,4; init=vinit(Wb)); L = sum(A * B)
                gA = similar(A3); gB = similar(Wb)
                for i in 1:2
                    gA[i,:,:] = ones(Float32,2,4) * permutedims(Wb[i,:,:])   # ∂L/∂A[i] = 1·Bᵀ
                    gB[i,:,:] = permutedims(A3[i,:,:]) * ones(Float32,2,4)    # ∂L/∂B[i] = Aᵀ·1
                end
                (E, JOLT.Tensor[∇(L;wrt=A), ∇(L;wrt=B)], (), [gA, gB])
            end,
            # ---- second-order (grads on grads) ----
            "2nd scalar d²(a²)=2"  => () -> (E, (a=Tensor(Var,Float32;init=Fill(3f0)); L=mul(a,a); gL=∇(L;wrt=a); JOLT.Tensor[L, gL, ∇(gL;wrt=a)]), (), [9f0, 6f0, 2f0]),
            "2nd vector d²Σa²"     => () -> (E, (a=V3([1,2,3]); L=sum(mul(a,a)); gL=∇(L;wrt=a); JOLT.Tensor[L, gL, ∇(gL;wrt=a, seed=Tensor(Const, ones(Float32,3)))]), (), [14f0, Float32[2,4,6], Float32[2,2,2]]),
            "2nd mixed d²Σab"      => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(a,b)); gLa=∇(L;wrt=a); JOLT.Tensor[gLa, ∇(gLa;wrt=b, seed=Tensor(Const, ones(Float32,3)))]), (), [Float32[4,5,6], ones(Float32,3)]),
        ]

        backends = Tuple{String,JOLT.IREEBackend}[("CPU", JOLT.IREE_CPU)]
        metal_ok = try
            new_session!(); xp = Tensor(1; name="x"); wp = Tensor(Var,1; init=Ones())
            fp, vp = compile(0, [xp], JOLT.Tensor[xp + wp]; backend=JOLT.IREE_METAL); fp(vp, Float32[1]); true
        catch; false end
        metal_ok ? push!(backends, ("Metal", JOLT.IREE_METAL)) :
                   @info "numeric Metal tests skipped — no Metal device / runtime available"

        for (bname, bk) in backends
            @testset "$bname" begin
                for (cname, thunk) in cases
                    @testset "$cname" begin
                        @test run1(bk, thunk)
                    end
                end
            end
        end

        # ---- symbolic (dynamic) dims through IREE, both backends — forward AND
        # gradients (incl. broadcasting a cotangent back into a dynamic axis via
        # dynamic_broadcast_in_dim), plus grads-on-grads through a dynamic graph. ----
        symcases = [
            # forward
            "sum over :B"    => () -> (x = Tensor(:B,4;name="x"); (JOLT.Tensor[x], JOLT.Tensor[sum(x)], (Float32[1 2 3 4; 5 6 7 8; 9 10 11 12],), [Float32(sum(1:12))])),
            "matmul :B×3"    => () -> (X = Tensor(:B,3;name="X"); W = Tensor(Var,3,4;init=Ones()); (JOLT.Tensor[X], JOLT.Tensor[X*W], (A2,), [A2 * ones(Float32,3,4)])),
            "reduce over :B" => () -> (x = Tensor(:B,4;name="x"); (JOLT.Tensor[x], JOLT.Tensor[reduce_sum(x,(1,))], (Float32[1 2 3 4; 5 6 7 8],), [vec(sum(Float32[1 2 3 4; 5 6 7 8]; dims=1))])),
            "two dyn args +" => () -> (x = Tensor(:B,4;name="x"); y = Tensor(:B,4;name="y"); (JOLT.Tensor[x,y], JOLT.Tensor[x+y], (Float32[1 2 3 4; 5 6 7 8], Float32[1 1 1 1; 2 2 2 2]), [Float32[2 3 4 5; 7 8 9 10]])),
            # gradients: cotangent broadcasts back over the dynamic axis
            "grad matmul :B (gW)" => () -> (X = Tensor(:B,3;name="X"); W = Tensor(Var,3,4;init=Ones()); L = sum(X*W); (JOLT.Tensor[X], JOLT.Tensor[∇(L;wrt=W)], (A2,), [permutedims(A2) * ones(Float32,2,4)])),
            "grad bias over :B (gb)" => () -> (x = Tensor(:B,4;name="x"); b = Tensor(Var,4;init=Ones()); L = sum(x .+ b); (JOLT.Tensor[x], JOLT.Tensor[∇(L;wrt=b)], (Float32[1 2 3 4; 5 6 7 8; 9 10 11 12],), [fill(3f0,4)])),
            # grads-on-grads THROUGH a dynamic graph: L=Σ(Xᵢⱼwⱼ)² ⇒ g1ⱼ=2wⱼΣᵢXᵢⱼ², g2ⱼ=2ΣᵢXᵢⱼ²
            "2nd-order through :B" => function ()
                Xv = Float32[1 2 3 4; 5 6 7 8]; cs = vec(sum(Xv.^2; dims=1))
                X = Tensor(:B,4;name="X"); w = Tensor(Var,4;init=Ones())
                y = X .* w; L = sum(mul(y,y)); g1 = ∇(L;wrt=w)
                (JOLT.Tensor[X], JOLT.Tensor[L, g1, ∇(sum(g1);wrt=w)], (Xv,), [sum(cs), 2 .* cs, 2 .* cs])
            end,
            # seeded VJP of a NON-scalar, dynamically-shaped loss via ones_like (no manual dynamic seed):
            # y=X.*w (shape (:B,4)); ones_like(y)ᵀ·Jᵥ = Σᵢ Xᵢₖ = colsum(X).
            "seeded VJP over :B (ones_like)" => () -> (X = Tensor(:B,4;name="X"); w = Tensor(Var,4;init=Ones());
                y = X .* w; (JOLT.Tensor[X], JOLT.Tensor[∇(y; wrt=w, seed=ones_like(y))], (Float32[1 2 3 4; 5 6 7 8],), [vec(sum(Float32[1 2 3 4; 5 6 7 8]; dims=1))])),
        ]
        @testset "symbolic dims" begin
            for (bname, bk) in backends
                @testset "$bname" begin
                    for (cn, th) in symcases
                        @testset "$cn" begin @test run1(bk, th) end
                    end
                end
            end
        end

        # ---- dtypes: f16 on all backends; f64 & integers CPU-only (Metal GPUs
        # have no f64). Declared dtypes are preserved (no silent f64→f32 demotion). ----
        @testset "dtypes" begin
            f16c = () -> begin
                A = Float16[1 2 3; 4 5 6]; B = Float16[6 5 4; 3 2 1]
                a = Tensor(Var,Float16,2,3;init=vinit(A)); b = Tensor(Var,Float16,2,3;init=vinit(B))
                (E, JOLT.Tensor[a+b, mul(a,b)], (), [A .+ B, A .* B])
            end
            for (bname, bk) in backends
                @testset "$bname f16" begin @test run1(bk, f16c) end
            end
            @testset "CPU f64 / i32 / i64" begin
                @test run1(JOLT.IREE_CPU, () -> begin A=Float64[1 2 3;4 5 6]; B=Float64[6 5 4;3 2 1]; a=Tensor(Var,Float64,2,3;init=vinit(A)); b=Tensor(Var,Float64,2,3;init=vinit(B)); (E, JOLT.Tensor[a+b, mul(a,b)], (), [A .+ B, A .* B]) end)          # f64 forward, eltype preserved
                @test run1(JOLT.IREE_CPU, () -> begin a=Tensor(Var,Float64,3;init=vinit(Float64[1,2,3])); b=Tensor(Var,Float64,3;init=vinit(Float64[4,5,6])); L=sum(mul(a,b)); (E, JOLT.Tensor[∇(L;wrt=a)], (), [Float64[4,5,6]]) end)                        # f64 gradient
                @test run1(JOLT.IREE_CPU, () -> begin A=Int32[1 2 3;4 5 6]; B=Int32[6 5 4;3 2 1]; a=Tensor(Var,Int32,2,3;init=vinit(A)); b=Tensor(Var,Int32,2,3;init=vinit(B)); (E, JOLT.Tensor[a+b, mul(a,b)], (), [A .+ B, A .* B]) end)                    # i32 (signless) forward
                @test run1(JOLT.IREE_CPU, () -> begin A=Int64[1 2 3;4 5 6]; B=Int64[6 5 4;3 2 1]; a=Tensor(Var,Int64,2,3;init=vinit(A)); b=Tensor(Var,Int64,2,3;init=vinit(B)); (E, JOLT.Tensor[a+b, mul(a,b)], (), [A .+ B, A .* B]) end)                    # i64 forward
            end
        end

        # ---- `vars` mutation is Julia-native and stays zero-copy: in-place edits
        # keep the SAME backing arena (so IREE sees them) and a reallocation is
        # rejected. All operations preserve `pointer(getdata(vars))`. ----
        @testset "vars mutation stays zero-copy (CPU)" begin
            new_session!()
            x = Tensor(2; name="x")
            namespace("p") do
                Tensor(Var, 2; init=Ones(),      name="w")
                Tensor(Var, 2; init=Fill(0.5f0), name="b")
            end
            fn, vars = compile(0, [x], JOLT.Tensor[mul(x + Tensor("variables","p","w"), Tensor("variables","p","b"))])
            p0 = pointer(JOLT.getdata(vars))
            @test fn(vars, Float32[10,20]) == Float32[5.5, 10.5]        # (10+1)·.5, (20+1)·.5
            vars.p.w .= Float32[100, 200]                              # subtree field .=
            @test pointer(JOLT.getdata(vars)) == p0
            @test fn(vars, Float32[10,20]) == Float32[55, 110]         # (110)·.5, (220)·.5
            vars.p.b[1] = 2f0                                          # element []=
            @test fn(vars, Float32[10,20]) == Float32[220, 110]        # (110)·2, (220)·.5
            JOLT.getdata(vars) .= 0f0                                  # whole-buffer .=
            @test pointer(JOLT.getdata(vars)) == p0
            @test fn(vars, Float32[10,20]) == Float32[0, 0]
            @test_throws Exception fn(vars .+ 0f0, Float32[10,20])     # a reallocated vars is rejected
        end

        # ---- a REAL Optimisers.jl step: update! mutates `vars` in place, keeps the
        # arena pointer (so it stays zero-copy), and IREE reads the updated weights. ----
        @testset "Optimisers step stays zero-copy (CPU)" begin
            new_session!()
            w = Tensor(Var, 3; init=vinit(Float32[3,4,5]))
            L = sum(mul(w, w)); gW = ∇(L; wrt=w)
            fn, vars = compile(0, JOLT.Tensor[], JOLT.Tensor[L, gW]; backend=JOLT.IREE_CPU)
            p0 = pointer(JOLT.getdata(vars))
            st = Optimisers.setup(Optimisers.Descent(0.1f0), vars)
            L0, g0 = fn(vars)                                          # L0 = 50, g0 = 2w = [6,8,10]
            st, vars = Optimisers.update!(st, vars, vec(g0))          # in-place descent step
            @test pointer(JOLT.getdata(vars)) == p0                    # arena preserved ⇒ still zero-copy
            L1, _ = fn(vars)                                           # IREE reads the updated weights
            @test L1 < L0                                              # the step decreased the loss
        end
    end
end
