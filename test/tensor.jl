# =====================================================================
# tensor.jl — the tensor value types and their construction surface.
#
# Focus: the handles themselves (concrete role types, traits, shape, host
# values, identity) and the internals they wrap (the role-specific payload, and
# the MLIR SSA value / type).
# =====================================================================

@testset "tensor" begin

    @testset "roles & aliases" begin
        # The four roles are concrete subtypes of AbstractTensor; Arg/Var/Const
        # are user-facing aliases for the three a user can construct.
        @test Arg   === Argument
        @test Var   === Variable
        @test Const === Constant
        @test Argument <: AbstractTensor && Variable <: AbstractTensor &&
              Constant <: AbstractTensor && Result <: AbstractTensor
        @test isconcretetype(Argument{Float32,2})

        # Result is exported (for dispatch / isa) but has NO public constructor.
        @test_throws MethodError Result()
        @test_throws MethodError Result(2, 3)
    end

    # Every constructor family resolves to the exact concrete type we expect —
    # element type, rank, and the role.
    @testset "construction: concrete types" begin
        new_session!()
        @test typeof(Arg())               === Argument{Float32,0}
        @test typeof(Arg(2, 3))           === Argument{Float32,2}
        @test typeof(Arg(Float64, 5))     === Argument{Float64,1}
        @test typeof(Arg(4))              === Argument{Float32,1}
        @test typeof(Arg(Float64, 2, 2))  === Argument{Float64,2}

        @test typeof(Var(2, 3))           === Variable{Float32,2}
        @test typeof(Var(Float32))        === Variable{Float32,0}
        @test typeof(Var(Ones(), 5)) === Variable{Float32,1}
        @test typeof(Var(Float64, 2, 2))  === Variable{Float64,2}

        @test typeof(Const([1f0, 2f0]))       === Constant{Float32,1}
        @test typeof(Const([1f0 2f0; 3f0 4f0])) === Constant{Float32,2}
        @test typeof(Const(3.0f0))            === Constant{Float32,0}
        @test typeof(Const(Float64, [1f0]))   === Constant{Float64,1}
    end

    # Symbolic dims: a Symbol in a shape slot becomes a polynomial dim (Poly);
    # a Poly expression can be passed directly. Rank counts them like any dim;
    # dtype/role are unaffected; Int and symbolic can be mixed.
    @testset "construction: symbolic dims" begin
        new_session!()
        @test typeof(Arg(:B, 10))      === Argument{Float32,2}
        @test typeof(Arg(Float64, :B)) === Argument{Float64,1}
        @test typeof(Arg(:B, 3))       === Argument{Float32,2}

        @test Arg(:B, 10).shape == (todim(:B), 10)    # Symbol -> Poly
        @test Arg(2, :M).shape[2] isa JOLT.Poly
        # same symbol -> same dim across tensors (the identity that makes checks provable)
        @test Arg(:B, 3).shape[1] == Arg(:B, 9).shape[1]
        @test Arg(:B, 3).shape[1] != Arg(:C, 9).shape[1]

        # derived dims go straight into shapes (e.g. a conv output)
        t = Arg(:B, fld(todim(:H), 16))
        @test t.shape == (todim(:B), fld(todim(:H), 16))
        @test ndims(t) == 2
    end

    # realize: substitute concrete sizes into a symbolic shape.
    @testset "realize on tensors" begin
        new_session!()
        t = Arg(:B, 3, fld(todim(:H), 16))
        @test realize(t, :B => 8, :H => 240) == (8, 3, 15)
        @test realize(Arg(4, 5)) == (4, 5)            # static needs no subs
        @test_throws ErrorException realize(t, :B => 8)  # :H missing
    end

    @testset "constants require a value" begin
        new_session!()
        @test_throws ErrorException Const()               # no value at all
        @test_throws ErrorException Const(2, 3)           # a shape, not a value
        @test_throws ErrorException Const(Float32, 2, 3)  # dtype + shape
    end

    @testset "traits: instance & type" begin
        new_session!()
        # role is the type now — check with isa
        @test Arg()          isa Argument
        @test Var(2)         isa Variable
        @test Const([1f0])   isa Constant

        @test eltype(Arg(Float64, 3)) == Float64
        @test ndims(Arg(2, 3, 4))     == 3

        # traits (and the role) read off the type directly, too
        @test eltype(Argument{Float64,1}) == Float64
        @test ndims(Argument{Float32,2})  == 2
        @test Variable{Float32,1} <: AbstractTensor{Float32,1}
    end

    @testset "shape, size, length, indexing" begin
        new_session!()
        t = Arg(Float32, 4, 5)
        @test size(t)    == (4, 5)
        @test size(t, 1) == 4
        @test size(t, 2) == 5
        @test size(t, 5) == 1               # trailing/out-of-range dim reads as 1 (Base semantics)
        @test length(t)  == 20
        @test ndims(t)   == 2

        sc = Arg()                          # scalar: prod(()) == 1
        @test size(sc)   == ()
        @test size(sc, 1) == 1              # scalar: any dim is 1, not a BoundsError
        @test length(sc) == 1

        dyn = Arg(Float32, 8, :N)
        @test size(dyn)    == (8, todim(:N))
        @test size(dyn, 2) == todim(:N)
        @test todim(:N) in size(dyn)
        @test_throws ErrorException length(dyn)   # undefined with a symbolic dim
        @test_throws ErrorException t[1]          # symbolic: no indexing
    end

    @testset "host values (Constant only)" begin
        new_session!()
        c = Const([1f0 2f0; 3f0 4f0])
        @test getvalue(c) == [1f0 2f0; 3f0 4f0]

        # construction copies the value — no aliasing of the caller's array
        src = [1f0, 2f0, 3f0]
        c2  = Const(src)
        src[1] = 99f0
        @test getvalue(c2) == [1f0, 2f0, 3f0]

        # no other role exposes a host value
        @test_throws ErrorException getvalue(Arg(2, 2))     # Argument
        @test_throws ErrorException getvalue(Var(2))        # Variable: realized at compile
    end

    @testset "identity, equality, copy" begin
        new_session!()
        a = Arg(2, 2)
        b = Arg(2, 2)
        @test a == a
        @test isequal(a, a)
        @test hash(a) == hash(a)
        @test a != b                        # distinct SSA values
        @test copy(a)     === a             # a tensor is just a handle
        @test deepcopy(a) === a
    end

    # The role model refuses invalid tensors up front, with a clear message.
    @testset "construction guards" begin
        new_session!()
        # only an Argument may be symbolic
        @test_throws ErrorException Var(:B, 3)
        @test_throws ErrorException Var(Float32, :B)
        # a Variable is shape + init, never a value (that's a Constant)
        @test_throws ErrorException Var([1f0, 2f0])
        @test_throws ErrorException Var(3.0f0)
        @test_throws ErrorException Var(Float64, [1f0])
        # a Result is produced by ops, never constructed
        @test_throws MethodError Result()
        @test_throws MethodError Result(2, 3)
        # a wrong-signature init (init is a Function positional) is caught at
        # construction, not deferred to compile. A non-Function in the init slot
        # can't be mistaken for one — it's just read as a dim (so no "not a
        # function" case exists anymore).
        @test_throws ErrorException Var(ones, 3)    # Base.ones: no (rng, T, …) method
        @test_throws ErrorException Var(Zeros, 3)   # forgot the parentheses
    end

    # Initializers are plain closures (rng, T, dims...) -> Array; a Variable
    # stores one and it is only *called* at compile time.
    @testset "initializers" begin
        rng() = JOLT.Random.MersenneTwister(0)
        @test Zeros()(rng(), Float32, 2, 3) == zeros(Float32, 2, 3)
        @test Ones()(rng(), Float64, 4)     == ones(Float64, 4)
        @test Fill(0.5)(rng(), Float32, 3)  == fill(0.5f0, 3)

        # random inits: right dtype/shape, and deterministic given the same rng
        r = RandN(0, 2)(rng(), Float32, 100)
        @test eltype(r) == Float32 && size(r) == (100,)
        @test RandN(0, 2)(rng(), Float32, 100) == r          # same seed -> same draw

        g = GlorotUniform()(rng(), Float32, 4, 5)
        @test eltype(g) == Float32 && size(g) == (4, 5)

        # rank-0 (scalar) inits must still return Array{T,0}, like the deterministic ones
        @test Zeros()(rng(), Float32)        isa Array{Float32,0}
        @test RandN()(rng(), Float32)        isa Array{Float32,0}
        @test Rand()(rng(), Float32)         isa Array{Float32,0}
        @test GlorotNormal()(rng(), Float32) isa Array{Float32,0}

        # a Variable stores the initializer function unevaluated
        new_session!()
        @test JOLT.initializer(Var(Ones(), 2)) isa Function

        # a user-supplied closure works just as well
        my = (rng, T, dims...) -> fill(T(7), dims...)
        @test my(rng(), Float32, 2) == fill(7f0, 2)
    end

    @testset "display" begin
        new_session!()
        @test sprint(show, Arg())               == "Argument{Float32,0}(scalar)"
        @test sprint(show, Arg(4, 5))           == "Argument{Float32,2}(4×5)"
        @test sprint(show, Arg(Float32, 8, :N)) == "Argument{Float32,2}(8×N)"
        @test sprint(show, Var(3))              == "Variable{Float32,1}(3)"
        # text/plain uses the same one-line form
        @test sprint(show, MIME("text/plain"), Arg(2, 2)) == "Argument{Float32,2}(2×2)"
    end

    # Each role names exactly the payload it needs (no shared union field), plus
    # the shape field.
    @testset "internals: role-specific payload" begin
        new_session!()
        @test fieldnames(Argument{Float32,2}) == (:value, :shape)   # Argument carries nothing extra
        @test fieldnames(Result{Float32,2})   == (:value, :shape)   # nor does a Result
        v = Var(2, 3)
        @test v.init isa Function                          # a Variable carries its initializer
        c = Const([1f0, 2f0])
        @test c.data == [1f0, 2f0]                         # a Constant carries its value
        @test getvalue(c) === c.data

        @test Arg(2, :M).shape == (2, todim(:M))           # shape field holds the symbolic dim
        @test Arg(2, :M).shape[2] isa JOLT.Poly
    end

    # The SSA value kind, and that the emitted MLIR type matches the handle.
    @testset "internals: SSA value & MLIR type" begin
        new_session!()
        @test Arg(2, 2).value isa JOLT.IR.Value
        @test JOLT.IR.is_block_arg(Arg(3, 3).value)        # argument
        @test JOLT.IR.is_block_arg(Var(2).value)           # variable
        @test JOLT.IR.is_op_res(Const([1f0]).value)        # constant is an op result

        @test mlir_ranked(Arg(Float32, 4, 5))         == (Float32, (4, 5))
        @test mlir_ranked(Arg(Float64, 2))            == (Float64, (2,))
        @test mlir_ranked(Arg())                      == (Float32, ())
        @test mlir_ranked(Arg(Float32, 8, :N))        == (Float32, (8, :dyn))   # symbolic -> ?
        @test mlir_ranked(Arg(:B, 10))                == (Float32, (:dyn, 10))
        @test mlir_ranked(Var(2))                     == (Float32, (2,))
        @test mlir_ranked(Const([1f0 2f0; 3f0 4f0]))  == (Float32, (2, 2))
    end

end
