# =====================================================================
# ops/gradients.jl — reverse-mode ∇, selection, second order, and the
# gradient-routing ops (stop_gradient, grad_reversal).
#
# Gradients are graph tensors (backward = more forward graph), so tests assert
# STRUCTURE — shapes, roles, tape growth, which tensors got a cotangent — not
# numeric values (those come at compile time).
# =====================================================================

@testset "gradients" begin

    # loss = a·b + a  (scalar). By hand: ∂/∂a = b+1, ∂/∂b = a.
    @testset "∇ first order & the return mirrors wrt" begin
        s = new_session!()
        a = Var(Float32)
        b = Var(Float32)
        loss = mul(a, b) + a

        # single tensor -> single gradient tensor
        ga = ∇(loss; wrt=a)
        @test ga isa AbstractTensor && size(ga) == () && ga isa Result

        # vector of tensors -> aligned vector
        g = ∇(loss; wrt=[a, b])
        @test g isa Vector{<:AbstractTensor} && length(g) == 2
        @test all(t -> size(t) == (), g)

        # scope prefix (Symbol / Vector{Symbol}) -> path-keyed Dict
        gd = ∇(loss; wrt=:variables)
        @test gd isa Dict
        @test sort(collect(keys(gd))) ==
              [(:variables, :default, :_1), (:variables, :default, :_2)]
        @test ∇(loss; wrt=[:variables]) |> keys |> collect |> sort ==
              sort(collect(keys(gd)))

        # default wrt is :variables
        @test keys(∇(loss)) == keys(gd)
    end

    @testset "gradient alias" begin
        new_session!()
        a = Var(Float32)
        @test gradient === ∇
        @test gradient(a + a; wrt=a) isa AbstractTensor
    end

    @testset "second order (differentiate a gradient)" begin
        new_session!()
        a = Var(Float32)
        b = Var(Float32)
        loss = mul(a, b) + a
        ga = ∇(loss; wrt=a)          # = b + 1  (a graph tensor)
        gg = ∇(ga; wrt=b)            # ∂²loss/∂a∂b = 1
        @test gg isa AbstractTensor && size(gg) == () && gg isa Result
        # the mixed partial reaches b but NOT a: ∂²loss/∂a² = 0 -> a unused -> zero const
        gaa = ∇(ga; wrt=a)
        @test gaa isa Constant
    end

    @testset "unused wrt → structural zero" begin
        new_session!()
        a = Var(Float32)
        z = Var(Float32)     # never used in the loss
        loss = mul(a, a)
        @test ∇(loss; wrt=z) isa Constant                 # ∂loss/∂z = 0
        # prefix collection stays complete: z still appears with a zero
        gd = ∇(loss; wrt=:variables)
        @test length(gd) == 2 && all(t -> t isa AbstractTensor, values(gd))
    end

    @testset "stop_gradient cuts the flow" begin
        s = new_session!()
        v = Var(Float32)
        d = stop_gradient(v)         # bound once
        @test size(d) == () && d isa Result
        loss = mul(d, d)             # loss = sg(v)²
        @test ∇(loss; wrt=v) isa Constant                 # gradient cut → zero
        # its vjp hands back a zero, not the incoming cotangent
        ȳ = Arg(Float32)
        cts = JOLT.vjp(JOLT.StopGradient(), ȳ, [v], d)
        @test length(cts) == 1 && cts[1] !== ȳ
    end

    @testset "grad_reversal flips the gradient" begin
        s = new_session!()
        x = Arg(2, 2)
        gr = grad_reversal(x)
        @test size(gr) == (2, 2) && gr isa Result
        ȳ = Arg(2, 2)
        n0 = length(s.tape)
        cts = JOLT.vjp(JOLT.GradReversal(), ȳ, [x], gr)
        @test length(cts) == 1
        @test s.tape[end][1].op isa JOLT.NegOp            # backward = neg(ȳ)
        @test s.tape[end][1].inputs == [ȳ]
        @test length(s.tape) == n0 + 1
    end

    # The headline: gradients w.r.t. inputs with DYNAMIC (symbolic) shapes.
    # No reduction op yet, so we use the vjp form (an explicit seed cotangent).
    @testset "gradients w.r.t. dynamic-shaped inputs" begin
        new_session!()
        x = Arg(:B, 4)            # symbolic first dim
        y = Arg(:B, 4)
        z = mul(x, y)                # (B, 4) — non-scalar
        gx = ∇(z; wrt=x, seed=Arg(:B, 4))
        @test size(gx) == (todim(:B), 4)                  # cotangent keeps the ? dim
        @test mlir_ranked(gx) == (Float32, (:dyn, 4))

        u = Arg(:B, 4)            # unused dynamic input → zeros via x-x
        gu = ∇(z; wrt=u, seed=Arg(:B, 4))
        @test size(gu) == (todim(:B), 4)

        # scalar-loss guard fires without a seed
        @test_throws ErrorException ∇(z; wrt=x)
        @test_throws ErrorException ∇(Arg(2, 3))
    end

    @testset "fan-out accumulates" begin
        s = new_session!()
        x = Var(Float32)
        g = ∇(mul(x, x); wrt=x)                 # ∂/∂x = 1·x + 1·x — two paths summed
        @test g isa Result
        @test s.tape[end][1].op isa JOLT.AddOp  # last emitted op is the accumulation add
    end

    @testset "∇ w.r.t. an Argument (not just Variables)" begin
        new_session!()
        p = Arg(Float32)                     # scalar argument
        @test ∇(mul(p, p); wrt=p) isa Result
    end

    @testset "selection & seed guards" begin
        new_session!()
        a = Var(Float32)
        loss = mul(a, a)
        @test_throws ErrorException ∇(loss; wrt=:variabels)   # typo'd role → no match
        @test_throws ErrorException ∇(loss; wrt=:params)      # a scope, not a role → no match
        @test_throws ErrorException ∇(loss; wrt=Symbol[])     # empty prefix selects nothing

        g = ∇(loss; wrt=AbstractTensor[])                     # empty tensor set
        @test g isa Vector{<:AbstractTensor} && isempty(g)    # ...still typed Vector{AbstractTensor}

        zns = mul(Arg(2, 3), Arg(2, 3))
        @test_throws ErrorException ∇(zns; wrt=a, seed=Arg(5))  # seed shape ≠ loss shape

        stale = loss
        new_session!()                                        # loss now belongs to a dead session
        @test_throws ErrorException ∇(stale; wrt=Var(Float32))
    end

    # The payoff of the reduce/broadcast layer: real scalar losses via sum, and
    # gradients that flow back through broadcasting (the bias reduces correctly).
    @testset "gradients through broadcasting & sum" begin
        new_session!()
        W = Var(3, 4)
        b = Var(4)                                            # bias, broadcast over rows
        d = W .+ b
        loss = sum(d .* d)                                    # scalar loss, no rank-0 var needed
        g = ∇(loss; wrt=[W, b])
        @test size(g[1]) == (3, 4)                            # ∂/∂W keeps W's shape
        @test size(g[2]) == (4,)                              # ∂/∂b reduced over the broadcast axis
        @test ∇(sum(W); wrt=W) isa Result

        # second order flows through broadcasting + reduce too
        p = Var(3); q = Var(1)
        gp = ∇(sum(p .+ q); wrt=p)
        @test size(∇(sum(gp .* p); wrt=q)) == (1,)
    end

end
