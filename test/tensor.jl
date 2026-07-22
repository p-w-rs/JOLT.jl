# =====================================================================
# tensor.jl — the Tensor value type and its construction surface.
#
# Focus: the handle itself (type params, traits, shape, host values, identity)
# and the internals it wraps (the DataBox, and the MLIR SSA value / type).
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
        @test typeof(Tensor(Var, [1f0, 2f0]))     === JOLT.Tensor{Float32,1,Variable}
        @test typeof(Tensor(Var, Float64, [1f0])) === JOLT.Tensor{Float64,1,Variable}

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

    @testset "host values (Variable only)" begin
        new_session!()
        v = Tensor(Var, [1f0 2f0; 3f0 4f0])
        @test getvalue(v) == [1f0 2f0; 3f0 4f0]

        setvalue!(v, [5f0 6f0; 7f0 8f0])
        @test getvalue(v) == [5f0 6f0; 7f0 8f0]

        sv = Tensor(Var, Float32)
        setvalue!(sv, 3.0f0)                # scalar overload
        @test getvalue(sv) == fill(3f0)

        @test_throws ErrorException setvalue!(v, [1f0, 2f0, 3f0])   # shape mismatch

        # only Variables own host data
        a = Tensor(2, 2)
        @test_throws ErrorException getvalue(a)
        @test_throws ErrorException setvalue!(a, [1f0 2f0; 3f0 4f0])
        @test_throws ErrorException getvalue(Tensor(Const, [1f0]))

        # a Variable given a non-array value gets a message about the VALUE,
        # not a bogus "Variable tensors are not updatable"
        err = try setvalue!(v, "nope"); nothing catch e; sprint(showerror, e) end
        @test occursin("String", err) && occursin("AbstractArray", err)
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

    @testset "duplicate (Variable only)" begin
        new_session!()
        v = Tensor(Var, [1f0, 2f0])
        d = duplicate(v)
        @test roleof(d)   == Variable
        @test size(d)     == size(v)
        @test getvalue(d) == getvalue(v)    # copies the current data...
        @test d != v                        # ...into a genuinely new SSA value
        @test d in session().argvars        # and a new block argument

        setvalue!(d, [9f0, 9f0])
        @test getvalue(v) == [1f0, 2f0]     # boxes are independent

        @test_throws ErrorException duplicate(Tensor(2, 2))        # non-Variable
        @test_throws ErrorException duplicate(Tensor(Const, [1f0]))
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

    # Poke at the fields directly: the data cell, its typing, and copy semantics.
    @testset "internals: fields & DataBox" begin
        new_session!()
        @test Tensor(2, 2).data === nothing            # arguments carry no box
        @test Tensor(Const, [1f0]).data === nothing    # neither do constants

        v = Tensor(Var, [1f0, 2f0])
        @test v.data isa JOLT.DataBox{Float32,1}
        @test getvalue(v) === v.data.array             # getvalue returns the box's array

        # construction copies host data — no aliasing of the caller's array
        src = [1f0, 2f0, 3f0]
        w = Tensor(Var, src)
        src[1] = 99f0
        @test getvalue(w) == [1f0, 2f0, 3f0]

        @test Tensor(2, :M).shape == (2, todim(:M))    # shape field holds the symbolic dim
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
        @test mlir_ranked(Tensor(Var, [1f0, 2f0]))           == (Float32, (2,))
        @test mlir_ranked(Tensor(Const, [1f0 2f0; 3f0 4f0])) == (Float32, (2, 2))
    end

end
