# =====================================================================
# tensor.jl — the Tensor value type and its construction surface.
#
# Focus: the handle itself (type params, traits, shape, host values, identity)
# and the internals it wraps (the payload, and the MLIR SSA value / type).
# =====================================================================

@testset "tensor" begin

    @testset "roles & sentinels" begin
        @test Arg   === Argument
        @test Var   === Variable
        @test Const === Constant
        @test JOLT.Res === JOLT.Result
        @test Argument isa TensorRole

        @test JOLT.DEFAULT_ELTYPE === Float32
        @test JOLT.DEFAULT_ROLE   === Argument
    end

    # Every constructor family resolves to the exact concrete type we expect —
    # element type, rank, and the role value parameter.
    @testset "construction: concrete types" begin
        new_session!()
        @test typeof(Tensor())                    === JOLT.Tensor{Float32,0,Argument}
        @test typeof(Tensor(2, 3))                === JOLT.Tensor{Float32,2,Argument}
        @test typeof(Tensor(Float64, 5))          === JOLT.Tensor{Float64,1,Argument}
        @test typeof(Tensor(Arg, 4))              === JOLT.Tensor{Float32,1,Argument}
        @test typeof(Tensor(Float64, Arg, 2, 2))  === JOLT.Tensor{Float64,2,Argument}

        @test typeof(Tensor(Var, 2, 3))           === JOLT.Tensor{Float32,2,Variable}
        @test typeof(Tensor(Var, Float32))        === JOLT.Tensor{Float32,0,Variable}
        @test typeof(Tensor(Var, 5; init=Ones()))  === JOLT.Tensor{Float32,1,Variable}
        @test typeof(Tensor(Var, Float64, 2, 2))   === JOLT.Tensor{Float64,2,Variable}

        @test typeof(Tensor(Const, [1f0, 2f0]))     === JOLT.Tensor{Float32,1,Constant}
        @test typeof(Tensor([1f0 2f0; 3f0 4f0]))    === JOLT.Tensor{Float32,2,Constant}
        @test typeof(Tensor(3.0f0))                 === JOLT.Tensor{Float32,0,Constant}
        @test typeof(Tensor(Const, Float64, [1f0])) === JOLT.Tensor{Float64,1,Constant}
    end

    # Symbolic dims: a Symbol in a shape slot becomes a polynomial dim (Poly);
    # a Poly expression can be passed directly. Rank counts them like any dim;
    # dtype/role are unaffected; Int and symbolic can be mixed.
    @testset "construction: symbolic dims" begin
        new_session!()
        @test typeof(Tensor(:B, 10))      === JOLT.Tensor{Float32,2,Argument}
        @test typeof(Tensor(Float64, :B)) === JOLT.Tensor{Float64,1,Argument}
        @test typeof(Tensor(Arg, :B, 3))  === JOLT.Tensor{Float32,2,Argument}

        @test Tensor(:B, 10).shape == (todim(:B), 10)    # Symbol -> Poly
        @test Tensor(2, :M).shape[2] isa JOLT.Poly
        # same symbol -> same dim across tensors (the identity that makes checks provable)
        @test Tensor(:B, 3).shape[1] == Tensor(:B, 9).shape[1]
        @test Tensor(:B, 3).shape[1] != Tensor(:C, 9).shape[1]

        # derived dims go straight into shapes (e.g. a conv output)
        t = Tensor(:B, fld(todim(:H), 16))
        @test t.shape == (todim(:B), fld(todim(:H), 16))
        @test ndims(t) == 2
    end

    # realize: substitute concrete sizes into a symbolic shape.
    @testset "realize on tensors" begin
        new_session!()
        t = Tensor(:B, 3, fld(todim(:H), 16))
        @test realize(t, :B => 8, :H => 240) == (8, 3, 15)
        @test realize(Tensor(4, 5)) == (4, 5)            # static needs no subs
        @test_throws ErrorException realize(t, :B => 8)  # :H missing
    end

    @testset "constants require a value" begin
        new_session!()
        @test_throws ErrorException Tensor(Const)                 # no value at all
        @test_throws ErrorException Tensor(Const, 2, 3)           # a shape, not a value
        @test_throws ErrorException Tensor(Const, Float32, 2, 3)  # dtype + shape
    end

    @testset "traits: instance & type" begin
        new_session!()
        @test roleof(Tensor())             == Argument
        @test roleof(Tensor(Var, 2))       == Variable
        @test roleof(Tensor(Const, [1f0])) == Constant

        @test eltype(Tensor(Float64, 3)) == Float64
        @test ndims(Tensor(2, 3, 4))     == 3

        # traits read off the type directly, too
        @test eltype(JOLT.Tensor{Float64,1,Argument}) == Float64
        @test ndims(JOLT.Tensor{Float32,2,Argument})  == 2
        @test roleof(JOLT.Tensor{Float32,1,Variable}) == Variable
    end

    @testset "shape, size, length, indexing" begin
        new_session!()
        t = Tensor(Float32, 4, 5)
        @test size(t)    == (4, 5)
        @test size(t, 1) == 4
        @test size(t, 2) == 5
        @test length(t)  == 20
        @test ndims(t)   == 2

        sc = Tensor()                       # scalar: prod(()) == 1
        @test size(sc)   == ()
        @test length(sc) == 1

        dyn = Tensor(Float32, 8, :N)
        @test size(dyn)    == (8, todim(:N))
        @test size(dyn, 2) == todim(:N)
        @test todim(:N) in size(dyn)
        @test_throws ErrorException length(dyn)   # undefined with a symbolic dim
        @test_throws ErrorException t[1]          # symbolic: no indexing
    end

    @testset "host values (Constant only)" begin
        new_session!()
        c = Tensor(Const, [1f0 2f0; 3f0 4f0])
        @test getvalue(c) == [1f0 2f0; 3f0 4f0]

        # construction copies the value — no aliasing of the caller's array
        src = [1f0, 2f0, 3f0]
        c2  = Tensor(Const, src)
        src[1] = 99f0
        @test getvalue(c2) == [1f0, 2f0, 3f0]

        # no other role exposes a host value
        @test_throws ErrorException getvalue(Tensor(2, 2))     # Argument
        @test_throws ErrorException getvalue(Tensor(Var, 2))   # Variable: realized at compile
    end

    @testset "identity, equality, copy" begin
        new_session!()
        a = Tensor(2, 2)
        b = Tensor(2, 2)
        @test a == a
        @test isequal(a, a)
        @test hash(a) == hash(a)
        @test a != b                        # distinct SSA values
        @test copy(a)     === a             # a Tensor is just a handle
        @test deepcopy(a) === a
    end

    # The role model refuses invalid tensors up front, with a clear message.
    @testset "construction guards" begin
        new_session!()
        # only an Argument may be symbolic
        @test_throws ErrorException Tensor(Var, :B, 3)
        @test_throws ErrorException Tensor(Var, Float32, :B)
        # a Variable is shape + init, never a value (that's a Constant)
        @test_throws ErrorException Tensor(Var, [1f0, 2f0])
        @test_throws ErrorException Tensor(Var, 3.0f0)
        @test_throws ErrorException Tensor(Var, Float64, [1f0])
        # a Result is produced by ops, never constructed
        @test_throws ErrorException Tensor(Res)
        @test_throws ErrorException Tensor(Result, 2, 3)
        # a wrong-signature init is caught at construction, not deferred to compile
        @test_throws ErrorException Tensor(Var, 3; init=ones)    # Base.ones: no (rng, T, …) method
        @test_throws ErrorException Tensor(Var, 3; init=Zeros)   # forgot the parentheses
        @test_throws ErrorException Tensor(Var, 3; init=42)      # not even a function
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

        # a Variable stores the initializer function unevaluated
        new_session!()
        @test JOLT.initializer(Tensor(Var, 2; init=Ones())) isa Function

        # a user-supplied closure works just as well
        my = (rng, T, dims...) -> fill(T(7), dims...)
        @test my(rng(), Float32, 2) == fill(7f0, 2)
    end

    @testset "display" begin
        new_session!()
        @test sprint(show, Tensor())                == "Tensor{Float32,0,Argument}(scalar)"
        @test sprint(show, Tensor(4, 5))            == "Tensor{Float32,2,Argument}(4×5)"
        @test sprint(show, Tensor(Float32, 8, :N)) == "Tensor{Float32,2,Argument}(8×N)"
        @test sprint(show, Tensor(Var, 3))          == "Tensor{Float32,1,Variable}(3)"
        # text/plain uses the same one-line form
        @test sprint(show, MIME("text/plain"), Tensor(2, 2)) == "Tensor{Float32,2,Argument}(2×2)"
    end

    # Poke at the payload field directly (per-role) and the shape field.
    @testset "internals: payload" begin
        new_session!()
        @test Tensor(2, 2).payload === nothing             # an Argument carries nothing
        v = Tensor(Var, 2, 3)
        @test v.payload isa Function                       # a Variable carries its initializer
        c = Tensor(Const, [1f0, 2f0])
        @test c.payload == [1f0, 2f0]                      # a Constant carries its value
        @test getvalue(c) === c.payload

        @test Tensor(2, :M).shape == (2, todim(:M))        # shape field holds the symbolic dim
        @test Tensor(2, :M).shape[2] isa JOLT.Poly
    end

    # The SSA value kind, and that the emitted MLIR type matches the handle.
    @testset "internals: SSA value & MLIR type" begin
        new_session!()
        @test Tensor(2, 2).value isa JOLT.IR.Value
        @test JOLT.IR.is_block_arg(Tensor(3, 3).value)        # argument
        @test JOLT.IR.is_block_arg(Tensor(Var, 2).value)      # variable
        @test JOLT.IR.is_op_res(Tensor(Const, [1f0]).value)   # constant is an op result

        @test mlir_ranked(Tensor(Float32, 4, 5))             == (Float32, (4, 5))
        @test mlir_ranked(Tensor(Float64, 2))                == (Float64, (2,))
        @test mlir_ranked(Tensor())                          == (Float32, ())
        @test mlir_ranked(Tensor(Float32, 8, :N))            == (Float32, (8, :dyn))   # symbolic -> ?
        @test mlir_ranked(Tensor(:B, 10))                    == (Float32, (:dyn, 10))
        @test mlir_ranked(Tensor(Var, 2))                    == (Float32, (2,))
        @test mlir_ranked(Tensor(Const, [1f0 2f0; 3f0 4f0])) == (Float32, (2, 2))
    end

end
