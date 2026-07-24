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
        E  = JOLT.AbstractTensor[]
        A2 = Float32[1 2 3; 4 5 6]
        B2 = Float32[6 5 4; 3 2 1]
        V3(x) = Var(vinit(Float32.(x)), 3)

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
            "fwd add"            => () -> (E, JOLT.AbstractTensor[Var(vinit(A2), 2,3) + Var(vinit(B2), 2,3)], (), [A2 .+ B2]),
            "fwd subtract"       => () -> (E, JOLT.AbstractTensor[Var(vinit(A2), 2,3) - Var(vinit(B2), 2,3)], (), [A2 .- B2]),
            "fwd mul"            => () -> (E, JOLT.AbstractTensor[mul(Var(vinit(A2), 2,3), Var(vinit(B2), 2,3))], (), [A2 .* B2]),
            "fwd negate"         => () -> (E, JOLT.AbstractTensor[-Var(vinit(A2), 2,3)], (), [-A2]),
            "fwd sum"            => () -> (E, JOLT.AbstractTensor[sum(Var(vinit(A2), 2,3))], (), [sum(A2)]),
            "fwd reduce_sum(2)"  => () -> (E, JOLT.AbstractTensor[reduce_sum(Var(vinit(A2), 2,3),(2,))], (), [dropdims(sum(A2;dims=2);dims=2)]),
            "fwd reshape(3,2)"   => () -> (E, JOLT.AbstractTensor[reshape(Var(vinit(A2), 2,3),3,2)], (), [reshape(A2,3,2)]),
            "fwd transpose"      => () -> (E, JOLT.AbstractTensor[permutedims(Var(vinit(A2), 2,3),(2,1))], (), [permutedims(A2)]),
            "fwd broadcast_to"   => () -> (E, JOLT.AbstractTensor[broadcast_to(Var(vinit(Float32[1;2;3;;]), 3,1),(3,4),[1,2])], (), [repeat(Float32[1;2;3;;],1,4)]),
            "fwd .+ scalar"      => () -> (E, JOLT.AbstractTensor[Var(vinit(A2), 2,3) .+ 1f0], (), [A2 .+ 1f0]),
            "fwd .+ leftpad vec" => () -> (E, JOLT.AbstractTensor[Var(vinit(A2), 2,3) .+ Var(vinit(Float32[10,20,30]), 3)], (), [A2 .+ reshape(Float32[10,20,30],1,3)]),
            "fwd matmul 2D"      => function ()
                Wm = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
                X = Arg(2,3; name="X"); W = Var(vinit(Wm), 3,4)
                (JOLT.AbstractTensor[X], JOLT.AbstractTensor[matmul(X, W)], (A2,), [A2 * Wm])
            end,
            "fwd matmul batched" => function ()
                A3 = permutedims(cat(Float32[1 0 1; 0 1 0], 2Float32[1 0 1; 0 1 0]; dims=3), (3,1,2))       # (2,2,3)
                Wb = permutedims(cat(Float32[1 0 0 1; 0 1 0 1; 0 0 1 1], Float32[1 0 0 1; 0 1 0 1; 0 0 1 1]; dims=3), (3,1,2))  # (2,3,4)
                A = Var(vinit(A3), 2,2,3); B = Var(vinit(Wb), 2,3,4)
                ref = similar(A3, 2,2,4); for i in 1:2; ref[i,:,:] = A3[i,:,:] * Wb[i,:,:]; end
                (E, JOLT.AbstractTensor[matmul(A, B)], (), [ref])
            end,
            # ---- first-order gradients ----
            "grad add"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(a+b);      JOLT.AbstractTensor[L, ∇(L;wrt=a), ∇(L;wrt=b)]), (), [21f0, ones(Float32,3), ones(Float32,3)]),
            "grad sub"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(a-b);      JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [ones(Float32,3), -ones(Float32,3)]),
            "grad mul"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(a,b)); JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [Float32[4,5,6], Float32[1,2,3]]),
            "grad transpose"     => () -> (E, (a=Var(vinit(A2), 2,3); t=permutedims(a,(2,1)); L=sum(mul(t,t)); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [2 .* A2]),
            "grad reshape"       => () -> (E, (a=Var(vinit(A2), 2,3); r=reshape(a,3,2); L=sum(mul(r,r)); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [2 .* A2]),
            "grad matmul"        => function ()
                X = Arg(2,3; name="X"); W = Var(Ones(), 3,4); L = sum(matmul(X, W))
                (JOLT.AbstractTensor[X], JOLT.AbstractTensor[∇(L;wrt=W)], (A2,), [permutedims(A2)*ones(Float32,2,4)])
            end,
            "grad stop_gradient" => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(stop_gradient(a),b)); JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [zeros(Float32,3), Float32[1,2,3]]),
            "grad grad_reversal" => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(grad_reversal(a),b)); JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [-Float32[4,5,6], Float32[1,2,3]]),
            "grad negate"        => () -> (E, (a=V3([1,2,3]); L=sum(-a); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [-ones(Float32,3)]),
            "grad reduce_sum"    => () -> (E, (a=Var(vinit(A2), 2,3); r=reduce_sum(a,(2,)); L=sum(mul(r,r)); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [2 .* repeat(sum(A2;dims=2),1,3)]),
            "grad broadcast_to"  => () -> (E, (a=Var(vinit(Float32[1;2;3;;]), 3,1); L=sum(broadcast_to(a,(3,4),[1,2])); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [fill(4f0,3,1)]),
            "grad .+ broadcast"  => () -> (E, (a=Var(vinit(A2), 2,3); b=Var(vinit(Float32[10,20,30]), 3); L=sum(a .+ b); JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [ones(Float32,2,3), fill(2f0,3)]),
            "grad matmul batched" => function ()
                A3 = permutedims(cat(Float32[1 0 1; 0 1 0], 2Float32[1 0 1; 0 1 0]; dims=3), (3,1,2))
                Wb = permutedims(cat(Float32[1 0 0 1; 0 1 0 1; 0 0 1 1], Float32[1 0 0 1; 0 1 0 1; 0 0 1 1]; dims=3), (3,1,2))
                A = Var(vinit(A3), 2,2,3); B = Var(vinit(Wb), 2,3,4); L = sum(matmul(A, B))
                gA = similar(A3); gB = similar(Wb)
                for i in 1:2
                    gA[i,:,:] = ones(Float32,2,4) * permutedims(Wb[i,:,:])   # ∂L/∂A[i] = 1·Bᵀ
                    gB[i,:,:] = permutedims(A3[i,:,:]) * ones(Float32,2,4)    # ∂L/∂B[i] = Aᵀ·1
                end
                (E, JOLT.AbstractTensor[∇(L;wrt=A), ∇(L;wrt=B)], (), [gA, gB])
            end,
            # ---- second-order (grads on grads) ----
            "2nd scalar d²(a²)=2"  => () -> (E, (a=Var(Float32, Fill(3f0)); L=mul(a,a); gL=∇(L;wrt=a); JOLT.AbstractTensor[L, gL, ∇(gL;wrt=a)]), (), [9f0, 6f0, 2f0]),
            "2nd vector d²Σa²"     => () -> (E, (a=V3([1,2,3]); L=sum(mul(a,a)); gL=∇(L;wrt=a); JOLT.AbstractTensor[L, gL, ∇(gL;wrt=a, seed=Const(ones(Float32,3)))]), (), [14f0, Float32[2,4,6], Float32[2,2,2]]),
            "2nd mixed d²Σab"      => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(mul(a,b)); gLa=∇(L;wrt=a); JOLT.AbstractTensor[gLa, ∇(gLa;wrt=b, seed=Const(ones(Float32,3)))]), (), [Float32[4,5,6], ones(Float32,3)]),
            # ---- elementwise divide, ⊙, and einsum / dot (forward) ----
            "fwd div"            => () -> (E, JOLT.AbstractTensor[Var(vinit(A2), 2,3) / Var(vinit(B2), 2,3)], (), [A2 ./ B2]),
            "fwd ./ leftpad vec" => () -> (E, JOLT.AbstractTensor[Var(vinit(A2), 2,3) ./ Var(vinit(Float32[1,2,3]), 3)], (), [A2 ./ reshape(Float32[1,2,3],1,3)]),
            "fwd ⊙ (== .*)"      => () -> (E, JOLT.AbstractTensor[Var(vinit(A2), 2,3) ⊙ Var(vinit(B2), 2,3)], (), [A2 .* B2]),
            "fwd einsum matmul"  => function ()
                Wm = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
                X = Arg(2,3; name="X"); W = Var(vinit(Wm), 3,4)
                (JOLT.AbstractTensor[X], JOLT.AbstractTensor[einsum("ik,kj->ij", X, W)], (A2,), [A2 * Wm])
            end,
            "fwd einsum batched" => function ()
                A3 = permutedims(cat(Float32[1 0 1; 0 1 0], 2Float32[1 0 1; 0 1 0]; dims=3), (3,1,2))
                Wb = permutedims(cat(Float32[1 0 0 1; 0 1 0 1; 0 0 1 1], Float32[1 0 0 1; 0 1 0 1; 0 0 1 1]; dims=3), (3,1,2))
                A = Var(vinit(A3), 2,2,3); B = Var(vinit(Wb), 2,3,4)
                ref = similar(A3, 2,2,4); for i in 1:2; ref[i,:,:] = A3[i,:,:] * Wb[i,:,:]; end
                (E, JOLT.AbstractTensor[einsum("bik,bkj->bij", A, B)], (), [ref])
            end,
            "fwd einsum dot"     => () -> (E, JOLT.AbstractTensor[einsum("i,i->", Var(vinit(Float32[1,2,3]),3), Var(vinit(Float32[4,5,6]),3))], (), [32f0]),
            "fwd einsum outer"   => () -> (E, JOLT.AbstractTensor[einsum("i,j->ij", Var(vinit(Float32[1,2,3]),3), Var(vinit(Float32[4,5]),2))], (), [Float32[1,2,3] .* Float32[4 5]]),
            "fwd dot"            => () -> (E, JOLT.AbstractTensor[dot(Var(vinit(Float32[1,2,3]),3), Var(vinit(Float32[4,5,6]),3))], (), [32f0]),
            # ---- gradients of the new ops ----
            "grad div"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=sum(a / b); JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [Float32[1/4,1/5,1/6], -Float32[1/16,2/25,3/36]]),
            "grad einsum matmul" => function ()
                X = Arg(2,3; name="X"); W = Var(Ones(), 3,4); L = sum(einsum("ik,kj->ij", X, W))
                (JOLT.AbstractTensor[X], JOLT.AbstractTensor[∇(L;wrt=W)], (A2,), [permutedims(A2)*ones(Float32,2,4)])
            end,
            "grad dot"           => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); L=dot(a,b); JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [Float32[4,5,6], Float32[1,2,3]]),
            # ---- second order THROUGH einsum: L=Σaᵢ²=dot(a,a) ⇒ g=2a, g²=2 ----
            "2nd via einsum dot" => () -> (E, (a=V3([1,2,3]); L=dot(a,a); gL=∇(L;wrt=a); JOLT.AbstractTensor[L, gL, ∇(gL;wrt=a, seed=Const(ones(Float32,3)))]), (), [14f0, Float32[2,4,6], Float32[2,2,2]]),
            # ---- new ops: activations, select, concatenate/slice/pad (fwd + grad) ----
            "fwd tanh"           => () -> (E, JOLT.AbstractTensor[tanh(Var(vinit(Float32[-0.5,0,0.8]),3))], (), [tanh.(Float32[-0.5,0,0.8])]),
            "fwd sqrt/rsqrt"     => () -> (E, JOLT.AbstractTensor[sqrt(Var(vinit(Float32[1,4,9]),3)), rsqrt(Var(vinit(Float32[1,4,0.25]),3))], (), [Float32[1,2,3], Float32[1,0.5,2]]),
            "fwd sigmoid/relu"   => () -> (E, JOLT.AbstractTensor[sigmoid(Var(vinit(Float32[0,100,-100]),3)), relu(Var(vinit(Float32[-1,2,-3]),3))], (), [Float32[0.5,1,0], Float32[0,2,0]]),
            "fwd select"         => () -> (X = Arg(;name="f"); (JOLT.AbstractTensor[X], JOLT.AbstractTensor[select(X .!= 0f0, Var(vinit(Float32[1,2,3]),3), Var(vinit(Float32[4,5,6]),3))], (fill(1f0),), [Float32[1,2,3]])),
            "fwd concatenate"    => () -> (E, JOLT.AbstractTensor[concatenate(Var(vinit(Float32[1,2]),2), Var(vinit(Float32[3,4,5]),3))], (), [Float32[1,2,3,4,5]]),
            "fwd slice/pad"      => () -> (E, JOLT.AbstractTensor[slice(Var(vinit(Float32[10,20,30,40,50]),5),2:4), pad(Var(vinit(Float32[7,8,9]),3);low=[1],high=[2])], (), [Float32[20,30,40], Float32[0,7,8,9,0,0]]),
            "grad tanh"          => () -> (E, (a=V3([-0.5,0,0.8]); L=sum(tanh(a)); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [1 .- tanh.(Float32[-0.5,0,0.8]).^2]),
            "grad relu"          => () -> (E, (a=V3([-1,2,-3]); L=sum(relu(a)); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [Float32[0,1,0]]),
            "grad sqrt"          => () -> (E, (a=V3([0.25,4,1]); L=sum(sqrt(a)); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [1f0 ./ (2 .* Float32[0.5,2,1])]),
            "grad concatenate"   => () -> (E, (a=V3([1,2,3]); b=V3([4,5,6]); c=concatenate(a,b); L=sum(mul(c,c)); JOLT.AbstractTensor[∇(L;wrt=a), ∇(L;wrt=b)]), (), [Float32[2,4,6], Float32[8,10,12]]),
            "grad slice"         => () -> (E, (a=Var(vinit(Float32[10,20,30,40,50]),5); s=slice(a,2:4); L=sum(mul(s,s)); JOLT.AbstractTensor[∇(L;wrt=a)]), (), [Float32[0,40,60,80,0]]),
        ]

        backends = Tuple{String,JOLT.IREEBackend}[("CPU", JOLT.IREE_CPU)]
        metal_ok = try
            new_session!(); xp = Arg(1; name="x"); wp = Var(Ones(), 1)
            fp, vp = compile(0, [xp], JOLT.AbstractTensor[xp + wp]; backend=JOLT.IREE_METAL); fp(vp, Float32[1]); true
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
            "sum over :B"    => () -> (x = Arg(:B,4;name="x"); (JOLT.AbstractTensor[x], JOLT.AbstractTensor[sum(x)], (Float32[1 2 3 4; 5 6 7 8; 9 10 11 12],), [Float32(sum(1:12))])),
            "matmul :B×3"    => () -> (X = Arg(:B,3;name="X"); W = Var(Ones(), 3,4); (JOLT.AbstractTensor[X], JOLT.AbstractTensor[matmul(X, W)], (A2,), [A2 * ones(Float32,3,4)])),
            "reduce over :B" => () -> (x = Arg(:B,4;name="x"); (JOLT.AbstractTensor[x], JOLT.AbstractTensor[reduce_sum(x,(1,))], (Float32[1 2 3 4; 5 6 7 8],), [vec(sum(Float32[1 2 3 4; 5 6 7 8]; dims=1))])),
            "two dyn args +" => () -> (x = Arg(:B,4;name="x"); y = Arg(:B,4;name="y"); (JOLT.AbstractTensor[x,y], JOLT.AbstractTensor[x+y], (Float32[1 2 3 4; 5 6 7 8], Float32[1 1 1 1; 2 2 2 2]), [Float32[2 3 4 5; 7 8 9 10]])),
            # gradients: cotangent broadcasts back over the dynamic axis
            "grad matmul :B (gW)" => () -> (X = Arg(:B,3;name="X"); W = Var(Ones(), 3,4); L = sum(matmul(X, W)); (JOLT.AbstractTensor[X], JOLT.AbstractTensor[∇(L;wrt=W)], (A2,), [permutedims(A2) * ones(Float32,2,4)])),
            "grad bias over :B (gb)" => () -> (x = Arg(:B,4;name="x"); b = Var(Ones(), 4); L = sum(x .+ b); (JOLT.AbstractTensor[x], JOLT.AbstractTensor[∇(L;wrt=b)], (Float32[1 2 3 4; 5 6 7 8; 9 10 11 12],), [fill(3f0,4)])),
            # grads-on-grads THROUGH a dynamic graph: L=Σ(Xᵢⱼwⱼ)² ⇒ g1ⱼ=2wⱼΣᵢXᵢⱼ², g2ⱼ=2ΣᵢXᵢⱼ²
            "2nd-order through :B" => function ()
                Xv = Float32[1 2 3 4; 5 6 7 8]; cs = vec(sum(Xv.^2; dims=1))
                X = Arg(:B,4;name="X"); w = Var(Ones(), 4)
                y = X .* w; L = sum(mul(y,y)); g1 = ∇(L;wrt=w)
                (JOLT.AbstractTensor[X], JOLT.AbstractTensor[L, g1, ∇(sum(g1);wrt=w)], (Xv,), [sum(cs), 2 .* cs, 2 .* cs])
            end,
            # seeded VJP of a NON-scalar, dynamically-shaped loss via ones_like (no manual dynamic seed):
            # y=X.*w (shape (:B,4)); ones_like(y)ᵀ·Jᵥ = Σᵢ Xᵢₖ = colsum(X).
            "seeded VJP over :B (ones_like)" => () -> (X = Arg(:B,4;name="X"); w = Var(Ones(), 4);
                y = X .* w; (JOLT.AbstractTensor[X], JOLT.AbstractTensor[∇(y; wrt=w, seed=ones_like(y))], (Float32[1 2 3 4; 5 6 7 8],), [vec(sum(Float32[1 2 3 4; 5 6 7 8]; dims=1))])),
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
                a = Var(Float16, vinit(A),2,3); b = Var(Float16, vinit(B),2,3)
                (E, JOLT.AbstractTensor[a+b, mul(a,b)], (), [A .+ B, A .* B])
            end
            for (bname, bk) in backends
                @testset "$bname f16" begin @test run1(bk, f16c) end
            end
            @testset "CPU f64 / i32 / i64" begin
                @test run1(JOLT.IREE_CPU, () -> begin A=Float64[1 2 3;4 5 6]; B=Float64[6 5 4;3 2 1]; a=Var(Float64, vinit(A),2,3); b=Var(Float64, vinit(B),2,3); (E, JOLT.AbstractTensor[a+b, mul(a,b)], (), [A .+ B, A .* B]) end)          # f64 forward, eltype preserved
                @test run1(JOLT.IREE_CPU, () -> begin a=Var(Float64, vinit(Float64[1,2,3]),3); b=Var(Float64, vinit(Float64[4,5,6]),3); L=sum(mul(a,b)); (E, JOLT.AbstractTensor[∇(L;wrt=a)], (), [Float64[4,5,6]]) end)                        # f64 gradient
                @test run1(JOLT.IREE_CPU, () -> begin A=Int32[1 2 3;4 5 6]; B=Int32[6 5 4;3 2 1]; a=Var(Int32, vinit(A),2,3); b=Var(Int32, vinit(B),2,3); (E, JOLT.AbstractTensor[a+b, mul(a,b)], (), [A .+ B, A .* B]) end)                    # i32 (signless) forward
                @test run1(JOLT.IREE_CPU, () -> begin A=Int64[1 2 3;4 5 6]; B=Int64[6 5 4;3 2 1]; a=Var(Int64, vinit(A),2,3); b=Var(Int64, vinit(B),2,3); (E, JOLT.AbstractTensor[a+b, mul(a,b)], (), [A .+ B, A .* B]) end)                    # i64 forward
            end
        end

        # ---- `vars` mutation is Julia-native and stays zero-copy: in-place edits
        # keep the SAME backing arena (so IREE sees them) and a reallocation is
        # rejected. All operations preserve `pointer(getdata(vars))`. ----
        @testset "vars mutation stays zero-copy (CPU)" begin
            new_session!()
            x = Arg(2; name="x")
            namespace("p") do
                Var(Ones(), 2; name="w")
                Var(Fill(0.5f0), 2; name="b")
            end
            fn, vars = compile(0, [x], JOLT.AbstractTensor[mul(x + getTensor("variables","p","w"), getTensor("variables","p","b"))])
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
            w = Var(vinit(Float32[3,4,5]), 3)
            L = sum(mul(w, w)); gW = ∇(L; wrt=w)
            fn, vars = compile(0, JOLT.AbstractTensor[], JOLT.AbstractTensor[L, gW]; backend=JOLT.IREE_CPU)
            p0 = pointer(JOLT.getdata(vars))
            st = Optimisers.setup(Optimisers.Descent(0.1f0), vars)
            L0, g0 = fn(vars)                                          # L0 = 50, g0 = 2w = [6,8,10]
            st, vars = Optimisers.update!(st, vars, vec(g0))          # in-place descent step
            @test pointer(JOLT.getdata(vars)) == p0                    # arena preserved ⇒ still zero-copy
            L1, _ = fn(vars)                                           # IREE reads the updated weights
            @test L1 < L0                                              # the step decreased the loss
        end

        # ---- stateful variables (assign!), train/test mode, and a PackedGrad
        # scope-selection driving a real Optimisers.update! on vars.params. ----
        @testset "stateful + mode + PackedGrad Optimisers (CPU)" begin
            # Hebbian: plastic W advances IN PLACE each call; deepcopy snapshots it
            new_session!()
            x = Arg(3; name="x")
            pushnamespace!("h"); W = Var(vinit(Float32[1,1,1]), 3; name="W"); popnamespace!()
            y = W .* x; assign!(W, W .+ x)
            fn, vars = compile(0, [x], JOLT.AbstractTensor[sum(y)])
            @test fn(vars, Float32[1,2,3]) ≈ 6f0  && collect(vars.h.W) == Float32[2,3,4]
            snap = deepcopy(vars)
            @test fn(vars, Float32[1,2,3]) ≈ 20f0 && collect(vars.h.W) == Float32[3,5,7]
            @test collect(snap.h.W) == Float32[2,3,4]                  # snapshot untouched

            # BatchNorm-style: running stat updates ONLY in train mode; frozen in test
            new_session!()
            train = Flag(true)
            pushnamespace!("bn"); mu = Var(vinit(Float32[0,0,0]), 3; name="mu"); popnamespace!()
            x = Arg(3; name="x")
            assign!(mu, train, 0.9f0 .* mu .+ 0.1f0 .* x)
            fn, vars = compile(0, [x], JOLT.AbstractTensor[x .- mu])
            trainmode!(vars); fn(vars, Float32[10,20,30])
            @test collect(vars.bn.mu) ≈ Float32[1,2,3]
            testmode!(vars); before = deepcopy(collect(vars.bn.mu)); fn(vars, Float32[10,20,30])
            @test collect(vars.bn.mu) == before

            # PackedGrad from a scope selection → ComponentArray → Optimisers.update!(vars.params)
            new_session!()
            pushnamespace!("params"); w = Var(vinit(Float32[3,4,5]), 3; name="w"); popnamespace!()
            L = sum(mul(w, w)); gs = ∇(L; wrt=[:variables, :params])
            fn, vars = compile(0, JOLT.AbstractTensor[], Any[L, gs])
            st = Optimisers.setup(Optimisers.Descent(0.1f0), vars.params)
            L0, g0 = fn(vars)
            @test g0 isa JOLT.ComponentArray && collect(g0.w) ≈ Float32[6,8,10]
            st, p = Optimisers.update!(st, vars.params, g0)            # functional; packed grad matches vars.params
            vars.params .= p                                           # write the step back into the shared arena (in place)
            L1, _ = fn(vars)
            @test L1 < L0
        end
    end
end
