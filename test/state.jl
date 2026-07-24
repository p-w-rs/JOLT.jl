# =====================================================================
# ops/state.jl — assign!, Flag, and train/test mode (construction-level;
# the in-place zero-copy behaviour end-to-end is verified in numeric.jl).
# =====================================================================

@testset "state (assign!, Flag, mode)" begin

    @testset "assign! records a state update (last-call-wins, shape-checked)" begin
        s = new_session!()
        v = Var(Ones(), 3)
        assign!(v, v .+ 1f0)
        @test haskey(s.assigns, v) && size(s.assigns[v]) == (3,)
        assign!(v, v .* 2f0)                              # last call wins → still one entry
        @test length(s.assigns) == 1
        @test_throws ErrorException assign!(Var(Ones(), 3), Arg(2))   # shape mismatch
    end

    @testset "Flag: a 0/1 scalar under variables/flags" begin
        s = new_session!()
        f1 = Flag()                                       # default on
        @test f1 isa Variable && size(f1) == ()
        @test haskey(s.names, (:variables, :flags, :_1))
        Flag(false, :training)
        @test haskey(s.names, (:variables, :flags, :training))
    end

    @testset "conditional assign! sugar builds a select over the flag" begin
        s = new_session!()
        mu = Var(Zeros(), 3); train = Flag()
        assign!(mu, train, 0.9f0 .* mu)
        @test haskey(s.assigns, mu)
        @test s.tape[end][1].op isa JOLT.SelectOp         # the recorded update is a select
    end

end
