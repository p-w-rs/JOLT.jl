# =====================================================================
# ops.jl — the Op/OpNode/apply machinery and the first two primitives.
#
# Focus: the three colocated aspects per op (outshape validates + infers,
# lower emits StableHLO, vjp builds backward graph), the session tape, the
# strict shape checking with facts-based overrides, and the StableHLO
# cross-check in push_op!.
# =====================================================================

# Deliberately-buggy ops, used to prove push_op!'s cross-checks actually fire
# (structs must be defined outside @testset blocks). `lower` receives the
# input TENSORS (not raw values) — same contract as the real ops.
struct BadShapeOp <: JOLT.Op end                     # claims the wrong output shape
JOLT.outshape(::BadShapeOp, sa::Tuple, sb::Tuple) = (99, 99)
JOLT.lower(::BadShapeOp, a::Tensor, b::Tensor) = JOLT.shlo.add(a.value, b.value)

struct BadRankOp <: JOLT.Op end                      # claims the wrong output rank
JOLT.outshape(::BadRankOp, sa::Tuple, sb::Tuple) = (sa..., 1)
JOLT.lower(::BadRankOp, a::Tensor, b::Tensor) = JOLT.shlo.add(a.value, b.value)

@testset "ops" begin

    # -----------------------------------------------------------------
    # apply: forward construction, result tensors, and the tape.
    # -----------------------------------------------------------------
    @testset "add: forward & tape" begin
        s = new_session!()
        x = Tensor(2, 3)
        y = Tensor(2, 3)
        z = x + y
        @test z isa JOLT.Tensor{Float32,2,JOLT.Result}
        @test size(z) == (2, 3)
        @test JOLT.IR.is_op_res(z.value)                 # a real StableHLO op result
        @test mlir_ranked(z) == (Float32, (2, 3))        # emitted IR type matches

        # the tape recorded exactly this application
        @test length(s.tape) == 1
        node, out = s.tape[1]
        @test node.op isa JOLT.AddOp
        @test node.inputs[1] === x && node.inputs[2] === y
        @test out === z
        @test node.shape == (2, 3)

        # results are registered anonymously under the :results role
        @test any(k -> k[1] === :results && startswith(string(k[end]), "_"), keys(s.names))
    end

    @testset "mul: forward & chaining" begin
        s = new_session!()
        x = Tensor(4)
        y = Tensor(4)
        z = mul(x, y)
        @test size(z) == (4,)
        w = z + x                                        # results feed later ops
        v = mul(w, z)
        @test size(v) == (4,)
        @test length(s.tape) == 3                        # emission order preserved
        @test s.tape[1][2] === z && s.tape[2][2] === w && s.tape[3][2] === v
        @test s.tape[2][1].inputs[1] === z               # w consumed z
    end

    @testset "scalars & dtypes" begin
        new_session!()
        a = Tensor()                                     # rank-0
        b = Tensor()
        c = a + b
        @test ndims(c) == 0 && size(c) == ()

        d = Tensor(Float64, 3)
        e = Tensor(Float64, 3)
        @test eltype(d + e) == Float64                   # dtype flows through

        f = Tensor(3)                                    # Float32
        err = try d + f; nothing catch e; sprint(showerror, e) end
        @test occursin("mixed element types", err)

        g = Tensor(Const, [1f0, 2f0, 3f0])               # constants participate
        k = Tensor(3)
        @test size(g + k) == (3,)
    end

    # -----------------------------------------------------------------
    # Strict shape checking: prove or reject; overrides make it provable.
    # -----------------------------------------------------------------
    @testset "shape rejection & overrides" begin
        new_session!()
        @test_throws ErrorException Tensor(2, 3) + Tensor(3, 2)   # static mismatch
        @test_throws ErrorException Tensor(2, 3) + Tensor(2, 3, 1) # rank mismatch

        # bare + is SAME-SHAPE ONLY — it never broadcasts. These will be the
        # canonical `.+` cases once the broadcast layer lands:
        @test_throws ErrorException Tensor(:N, 10) + Tensor(10)   # (N,10) ⊕ (10,)
        @test_throws ErrorException Tensor(:N, 1) + Tensor(:N)    # (N,1) ⊕ (N,)
        @test_throws ErrorException Tensor(2, 1) + Tensor(1, 2)   # even fully static

        # same symbol: provably equal, builds
        a = Tensor(:B, 4)
        b = Tensor(:B, 4)
        c = a + b
        @test size(c) == (todim(:B), 4)                  # symbolic dim survives inference
        @test mlir_ranked(c) == (Float32, (:dyn, 4))     # ...but lowers to ?

        # different symbols: rejected until declared equal — the LATE override
        p = Tensor(:U, 4)
        q = Tensor(:V, 4)
        err = try p + q; nothing catch e; sprint(showerror, e) end
        @test occursin("differ at dim", err) && occursin("same_dim!", err)
        same_dim!(:U, :V)                                # declared mid-construction
        r = p + q
        @test size(r) == (todim(:U), 4)                  # original symbol kept

        # derived dims: provable equality builds, unprovable rejects
        new_session!()
        m = Tensor(:H, 8)
        n = Tensor(fld(todim(:H), 2), 8)
        o = Tensor(fld(todim(:H), 2), 8)
        @test size(n + o) == (fld(todim(:H), 2), 8)
        @test_throws ErrorException m + n
    end

    # -----------------------------------------------------------------
    # Facts are per-session: an override in one session doesn't leak.
    # -----------------------------------------------------------------
    @testset "facts isolation across sessions" begin
        new_session!()
        same_dim!(:B, :D)
        @test dims_equal(todim(:B), todim(:D))
        new_session!()                                   # fresh facts
        @test !dims_equal(todim(:B), todim(:D))
        @test_throws ErrorException Tensor(:B, 2) + Tensor(:D, 2)
    end

    # -----------------------------------------------------------------
    # All roles participate in ops: Arguments, Variables, Constants.
    # -----------------------------------------------------------------
    @testset "mixed input roles" begin
        new_session!()
        x = Tensor(3)                                    # Argument
        v = Tensor(Var, 3)                               # Variable
        c = Tensor(Const, [4f0, 5f0, 6f0])               # Constant
        y = mul(x + v, c)                                # all three in one chain
        @test roleof(y) == Result
        @test size(y) == (3,)
        @test length(session().tape) == 2
    end

    # -----------------------------------------------------------------
    # A rejected op must leave ZERO partial state: nothing on the tape,
    # nothing in the registry, nothing appended to the block.
    # -----------------------------------------------------------------
    @testset "rejected op leaves no state" begin
        s = new_session!()
        x = Tensor(2, 3)
        y = Tensor(3, 2)
        names_before = length(s.names)
        @test_throws ErrorException x + y                # shape mismatch
        @test isempty(s.tape)
        @test length(s.names) == names_before
        z = x + Tensor(2, 3)                             # session still fully usable
        @test size(z) == (2, 3)
    end

    # -----------------------------------------------------------------
    # push_op!'s StableHLO cross-check fires on a buggy shape rule — and,
    # because validation precedes the append, leaves the block clean.
    # -----------------------------------------------------------------
    @testset "StableHLO cross-check catches bad shape rules" begin
        s = new_session!()
        x = Tensor(2, 3)
        y = Tensor(2, 3)
        err = try JOLT.apply(BadShapeOp(), x, y); nothing catch e; sprint(showerror, e) end
        @test occursin("shape rule bug", err) && occursin("BadShapeOp", err)
        err = try JOLT.apply(BadRankOp(), x, y); nothing catch e; sprint(showerror, e) end
        @test occursin("rank", err)
        @test isempty(s.tape)                            # neither op got recorded
        z = x + y                                        # block is still valid
        @test size(z) == (2, 3)
    end

    # -----------------------------------------------------------------
    # Tensors are bound to their session: mixing sessions is rejected at
    # the op, with a message that says so — not deferred to MLIR.
    # -----------------------------------------------------------------
    @testset "cross-session tensors rejected" begin
        new_session!()
        a1 = Tensor(Float32, 2)
        new_session!()
        b2 = Tensor(Float32, 2)
        err = try a1 + b2; nothing catch e; sprint(showerror, e) end
        @test occursin("different session", err)
        @test isempty(session().tape)                    # nothing recorded
        @test size(b2 + Tensor(Float32, 2)) == (2,)      # same-session still fine
    end

    # -----------------------------------------------------------------
    # Invalid concrete dims are rejected at construction, not deferred
    # into invalid MLIR types.
    # -----------------------------------------------------------------
    @testset "negative dims rejected" begin
        new_session!()
        @test_throws ErrorException Tensor(-3)
        @test_throws ErrorException Tensor(-3, 4)
        @test_throws ErrorException Tensor(Float64, 2, -1)
        @test size(Tensor(0, 4)) == (0, 4)               # zero-sized tensors are legal
    end

    # -----------------------------------------------------------------
    # vjp: each primitive's backward rule builds real graph nodes.
    # -----------------------------------------------------------------
    @testset "vjp rules" begin
        s = new_session!()
        x = Tensor(2, 2)
        y = Tensor(2, 2)
        z = mul(x, y)
        ȳ = Tensor(2, 2)                                # stand-in cotangent seed

        # add: passes the cotangent straight through — no new ops
        n0 = length(s.tape)
        cts = JOLT.vjp(JOLT.AddOp(), ȳ, [x, y], z)
        @test cts === (ȳ, ȳ)
        @test length(s.tape) == n0                       # nothing emitted

        # mul: x̄ = ȳ·y, ȳx = ȳ·x — two NEW mul nodes on the same tape,
        # each reading the *other* forward input
        cts = JOLT.vjp(JOLT.MulOp(), ȳ, [x, y], z)
        @test length(s.tape) == n0 + 2
        @test all(t -> size(t) == (2, 2), cts)
        @test s.tape[end-1][1].inputs == [ȳ, y]          # x̄ reads y
        @test s.tape[end][1].inputs == [ȳ, x]            # ȳx reads x
    end

    # -----------------------------------------------------------------
    # Symbolic-shaped ops keep symbolic vjps (backward = more forward).
    # -----------------------------------------------------------------
    @testset "vjp with symbolic shapes" begin
        new_session!()
        x = Tensor(:B, 8)
        y = Tensor(:B, 8)
        z = mul(x, y)
        ȳ = Tensor(:B, 8)
        cts = JOLT.vjp(JOLT.MulOp(), ȳ, [x, y], z)
        @test all(t -> size(t) == (todim(:B), 8), cts)
    end

    # -----------------------------------------------------------------
    # subtract & negate: forward, same-shape rule, and vjps.
    #   d(a-b) = (ȳ, -ȳ)      d(-a) = -ȳ
    # -----------------------------------------------------------------
    @testset "subtract & negate" begin
        s = new_session!()
        x = Tensor(2, 3)
        y = Tensor(2, 3)
        @test size(x - y) == (2, 3)
        @test size(-x) == (2, 3)                             # unary minus = negate
        @test_throws ErrorException Tensor(2, 3) - Tensor(3, 2)   # same-shape only

        # sub vjp: (ȳ, -ȳ) — the second cotangent is a fresh NegOp node
        ȳ = Tensor(2, 3)
        n0 = length(s.tape)
        cts = JOLT.vjp(JOLT.SubOp(), ȳ, [x, y], x - y)
        @test cts[1] === ȳ
        @test roleof(cts[2]) == Result && length(s.tape) == n0 + 2  # (x-y) + neg(ȳ)

        # neg vjp: (-ȳ,)
        cts = JOLT.vjp(JOLT.NegOp(), ȳ, [x], -x)
        @test length(cts) == 1 && size(cts[1]) == (2, 3)
    end

end
