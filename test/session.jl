# =====================================================================
# session.jl — the Session, its globals, name registry, and IR builders.
#
# Focus: session install/swap/reset, the MLIR context activation that rides
# along with it, the name registry / namespaces, and the builder side effects
# on the block and argvars.
# =====================================================================

@testset "session" begin

    @testset "lifecycle" begin
        s = Session()                       # constructed, not installed
        @test s.mod === nothing
        @test isempty(s.argvars) && isempty(s.names) && isempty(s.outputs)
        @test isempty(s.tape)               # no ops emitted yet
        @test isempty(s.facts.decls)        # no constraints declared yet
        @test isempty(s.anon)               # no anonymous names minted yet

        s1 = new_session!()
        @test session() === s1
        @test sprint(show, s1) == "Session(0 inputs, 0 named, 0 outputs)"
        Tensor()                            # one anonymous scalar argument
        @test sprint(show, s1) == "Session(1 inputs, 1 named, 0 outputs)"

        s2 = new_session!()                 # fresh, replaces s1
        @test session() === s2 && s2 !== s1
        @test isempty(s2.argvars)

        # with_session runs f against a scratch session, then restores.
        main    = session()
        scratch = Session()
        n = with_session(scratch) do
            @test session() === scratch
            Tensor(2, 3)
            length(scratch.argvars)
        end
        @test n == 1
        @test session() === main
        @test isempty(main.argvars)         # main was untouched

        # session! re-installs an existing session with its state intact.
        a = new_session!()
        Tensor(4)
        b = new_session!()
        @test session() === b
        session!(a)
        @test session() === a
        @test length(a.argvars) == 1

        # reset drops the current session; next session() builds a fresh one.
        new_session!()
        Tensor(4)
        reset_session!()
        @test JOLT._SESSION[] === nothing
        @test isempty(session().argvars)
    end

    # current_facts always points at the ACTIVE session's constraint store.
    @testset "facts follow the session" begin
        s1 = new_session!()
        @test current_facts() === s1.facts
        same_dim!(:B, :D)
        @test dims_equal(todim(:B), todim(:D))

        scratch = Session()
        with_session(scratch) do
            @test current_facts() === scratch.facts
            @test !dims_equal(todim(:B), todim(:D))   # scratch has its own facts
        end
        @test current_facts() === s1.facts
        @test dims_equal(todim(:B), todim(:D))        # s1's declaration intact

        new_session!()                                # fresh session, fresh facts
        @test !dims_equal(todim(:B), todim(:D))
    end

    # The session-facts convenience surface (no explicit Facts argument):
    # every query and mutator routes through current_facts().
    @testset "facts convenience API" begin
        new_session!()
        B, D, H = todim(:B), todim(:D), todim(:H)

        @test !dims_equal(B, D)
        same_dim!(:B, :D)
        @test dims_equal(B, D)

        pin!(:P, 1)
        @test canon(todim(:P)) === 1
        @test JOLT.broadcast_dim(B, todim(:P)) == B      # pinned-to-1 stretches

        divisible!(:H, 16)
        @test provably_divisible(H, 16)
        @test dims_equal(fld(H, 16) * 16, H)
        @test JOLT.contract_dim(fld(H, 16) * 16, H) === nothing

        bound!(:T; lo = 2, hi = 512)
        @test provably_ge(todim(:T), 2)

        @test JOLT.broadcast_shapes((B, 1), (1, 10)) == (B, 10)

        @test check_facts(Dict(:B => 4, :D => 4, :P => 1, :H => 32, :T => 8)) === nothing
        @test_throws ErrorException check_facts(Dict(:B => 4, :D => 5, :P => 1, :H => 32, :T => 8))
    end

    # Session swaps INSIDE a with_session block keep the context stack sane.
    @testset "session! inside with_session" begin
        outer = new_session!()
        scratch = Session()
        inner = with_session(scratch) do
            new_session!()                               # swap mid-block
            Tensor(2)
            session()
        end
        @test session() === outer                        # outer restored regardless
        @test JOLT.IR.current_context() == outer.context
        @test length(inner.argvars) == 1

        # ...and an exception inside the block still restores the outer session
        @test_throws ErrorException with_session(Session()) do
            error("boom")
        end
        @test session() === outer
        @test JOLT.IR.current_context() == outer.context
    end

    # The active session and the active MLIR context move together.
    @testset "context activation (internal)" begin
        s = new_session!()
        @test JOLT.IR.has_context()
        @test JOLT.IR.current_context() == s.context

        s2 = new_session!()
        @test JOLT.IR.current_context() == s2.context
        @test JOLT.IR.current_context() != s.context

        scratch = Session()
        with_session(scratch) do
            @test JOLT.IR.current_context() == scratch.context
        end
        @test JOLT.IR.current_context() == s2.context     # restored after the block

        reset_session!()
        @test !JOLT.IR.has_context()                      # stack emptied
        s3 = session()
        @test JOLT.IR.has_context()
        @test JOLT.IR.current_context() == s3.context     # rebuilt + reactivated
    end

    @testset "default dtype" begin
        @test default_dtype() == Float32
        try
            default_dtype!(Float64)
            @test default_dtype() == Float64
            new_session!()
            @test eltype(Tensor(3))      == Float64     # arg picks up the default
            @test eltype(Tensor(Var, 2)) == Float64     # so does a variable
        finally
            default_dtype!(Float32)                     # don't leak into later sets
        end
        @test default_dtype() == Float32
    end

    @testset "name registry" begin
        s = new_session!()
        @test isempty(s.anon)

        Tensor(2; name="x")                 # named: anon counter untouched
        @test s.anon[:arguments][] == 0
        @test haskey(s.names, (:arguments, :default, :x))
        @test length(s.names) == 1

        Tensor()                            # anonymous: counter advances
        @test s.anon[:arguments][] == 1
        @test haskey(s.names, (:arguments, :default, :_1))

        # counters are PER-ROLE (per tensor type): each starts fresh at _1
        s = new_session!()
        Tensor(Var, 2)                      # (:variables, :default, :_1)
        Tensor(2, 2)                        # (:arguments, :default, :_1)
        Tensor(Const, [1f0])                # (:constants, :default, :_1)
        @test s.anon[:variables][]  == 1
        @test s.anon[:arguments][]  == 1
        @test s.anon[:constants][]  == 1
        @test haskey(s.names, (:variables, :default, :_1))
        @test haskey(s.names, (:arguments, :default, :_1))
        @test haskey(s.names, (:constants, :default, :_1))

        # duplicate names within a scope are rejected; misses error
        new_session!()
        Tensor(2; name="dup")
        @test_throws ErrorException Tensor(2; name="dup")
        @test_throws ErrorException JOLT.lookup((:missing,))

        # string lookup returns the very same handle
        new_session!()
        W = Tensor(Var, 2, 2; name="W")
        @test JOLT.lookup((:variables, :default, :W)) === W
        @test Tensor("variables", "default", "W") === W
    end

    @testset "namespaces" begin
        new_session!()
        @test JOLT.effective_scope() == (:default,)
        w = namespace("Dense") do
            @test JOLT.effective_scope() == (:Dense,)
            inner = namespace("Layer") do
                @test JOLT.effective_scope() == (:Dense, :Layer)
                Tensor(Var, 2; name="w")
            end
            @test JOLT.effective_scope() == (:Dense,)   # inner scope popped
            inner
        end
        @test JOLT.effective_scope() == (:default,)     # outer scope popped
        @test JOLT.lookup((:variables, :Dense, :Layer, :w)) === w
        @test Tensor("variables", "Dense", "Layer", "w") === w

        # anonymous names are qualified by the current scope, too
        new_session!()
        namespace("A") do
            Tensor(Var, 2)
        end
        @test haskey(session().names, (:variables, :A, :_1))
    end

    # Declaring args/vars appends block arguments in declaration order;
    # constants don't touch the block or the input list.
    @testset "builders: block args track argvars (internal)" begin
        s = new_session!()
        @test JOLT.IR.nargs(s.block) == 0

        a = Tensor(3, 4)
        v = Tensor(Var, 2)
        @test JOLT.IR.nargs(s.block) == 2
        @test length(s.argvars) == 2
        @test s.argvars[1] === a && s.argvars[2] === v
        @test JOLT.IR.argument(s.block, 1) == a.value     # order preserved
        @test JOLT.IR.argument(s.block, 2) == v.value

        Tensor(Const, [1f0])                # a constant is neither
        @test JOLT.IR.nargs(s.block) == 2
        @test length(s.argvars) == 2

        sy = Tensor(:B, 4)                  # a symbolic-dim arg is still a block arg
        @test JOLT.IR.nargs(s.block) == 3
        @test s.argvars[3] === sy
    end

    @testset "mlir_type builder (internal)" begin
        new_session!()                      # needs an active context
        ty = JOLT.mlir_type(Float32, (2, 3))
        @test JOLT.IR.hasrank(ty)
        @test JOLT.IR.julia_type(JOLT.IR.eltype(ty)) == Float32
        @test JOLT.IR.ndims(ty) == 2
        @test Int(JOLT.IR.size(ty, 1)) == 2
        @test Int(JOLT.IR.size(ty, 2)) == 3

        dyn = JOLT.mlir_type(Float32, (8, todim(:N)))
        @test !JOLT.IR.isdynsize(JOLT.IR.size(dyn, 1))   # static dim stays
        @test JOLT.IR.isdynsize(JOLT.IR.size(dyn, 2))    # symbolic dim -> ?
    end

end
