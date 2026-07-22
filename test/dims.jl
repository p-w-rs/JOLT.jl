# =====================================================================
# dims.jl — the symbolic dimension engine.
#
# Pure Julia: every testset here uses an explicit `JOLT.Facts()` store, so
# no session or MLIR context is required (the one exception, dim_to_mlir /
# mlir_ok, only calls the context-free dynsize sentinel). Covers the
# polynomial normal form, fld/mod reduction, max/min, realize, the
# constraint store (equalities, rules, divisibility, bounds), the two
# TVM-style analyzers (ModularSet, ConstIntBound), the strict broadcast /
# contract queries, and the run-time check_facts gate.
# =====================================================================

@testset "dims" begin

    # Shared symbolic dims (todim is pure; scoped to this testset).
    H = todim(:H); W = todim(:W); B = todim(:B)
    D = todim(:D); T = todim(:T); h = todim(:h)

    # -----------------------------------------------------------------
    # Vocabulary: Int stays Int, Symbol -> variable poly, Dim passthrough.
    # -----------------------------------------------------------------
    @testset "todim & the Dim union" begin
        @test todim(10) === 10
        @test todim(Int32(3)) === 3                     # any Integer -> Int
        @test todim(true) === 1                         # Bool is an Integer too
        @test todim(:B) isa JOLT.Poly
        @test todim(:B) == B                            # same symbol -> equal poly
        @test todim(B) === B                            # Dim passes through untouched
        @test todim(7) isa JOLT.Dim && B isa JOLT.Dim

        # internal structure: :B is the polynomial 1·(B¹)
        m, c = only(B.terms)
        @test c == 1
        a, e = only(m.powers)
        @test a == JOLT.DimVar(:B) && e == 1
    end

    # -----------------------------------------------------------------
    # Normalization invariants: nothing-symbolic collapses to Int; zero
    # coefficients are pruned.
    # -----------------------------------------------------------------
    @testset "normal form invariants" begin
        @test H - H === 0                               # full cancellation -> Int zero
        @test (H + 2) - H === 2                         # constant residue -> Int
        @test H * 0 === 0                               # scale by zero -> Int zero
        @test 0 * H === 0
        @test (H + 1) - 1 == H                          # back to a bare variable
        @test H^0 === 1                                 # empty product -> Int one
        @test H + 0 == H && 0 + H == H && H - 0 == H && 1 * H == H && H * 1 == H
    end

    # -----------------------------------------------------------------
    # Arithmetic & algebraic identities — all decided by Dict equality on
    # the normal form, no rewrite rules involved.
    # -----------------------------------------------------------------
    @testset "arithmetic & identities" begin
        @test H + W == W + H                            # commutativity (+)
        @test H * W == W * H                            # commutativity (*)
        @test (H + W) + B == H + (W + B)                # associativity
        @test 2 * (H + W) == 2H + 2W                    # distribution
        @test H + H == 2H                               # like terms merge
        @test H + 2H + 3H == 6H
        @test (H + 1) * (W + 1) == H * W + H + W + 1     # full expansion
        @test (H + 1) * (H - 1) == H * H - 1             # difference of squares (cancellation)
        @test (H - 1) * (W + 1) == W * H + H - W - 1     # the case that forced polynomials
        @test H^3 == H * H * H                          # powers are repeated products
        @test (2H)^2 == 4 * H^2
        @test -(H - W) == W - H                         # unary negation
        @test +H == H
        @test_throws ErrorException H^-1                # negative powers rejected

        # hashing agrees with equality (safe as Dict keys)
        @test hash(H + W) == hash(W + H)
        @test hash((H + 1) * (H - 1)) == hash(H^2 - 1)
        d = Dict((H + W) => 1)
        @test d[W + H] == 1

        # non-Int Integer widths funnel through
        @test H + Int32(3) == H + 3
        @test UInt8(2) * H == 2H
        @test H - true == H - 1

        # symbolic never structurally equals concrete (sound: unknown ≠ known)
        @test B != 4 && 4 != B
        @test (H - H + 5) === 5                         # ...unless it collapses to Int
    end

    # -----------------------------------------------------------------
    # fld / mod: exact reduction iff the variable part is a multiple of k
    # (identity ⌊(k·q + c)/k⌋ = q + ⌊c/k⌋); everything else stays opaque.
    # -----------------------------------------------------------------
    @testset "fld & mod" begin
        @test fld(16h, 16) == h                         # exact
        @test fld(16h, 16) * 16 == 16h                  # patchify round-trip, structurally
        @test fld(16h + 8W, 8) == 2h + W                # mixed coefficients, all divisible
        @test fld(16h - 5, 16) == h - 1                 # negative constant floors correctly
        @test mod(16h, 16) === 0
        @test mod(16h + 3, 16) === 3
        @test mod(16h + 20, 16) === 4
        @test mod(16h - 5, 16) === 11                   # mod of negative constant

        @test fld(H, 16) isa JOLT.Poly                  # not provably divisible -> opaque
        @test fld(H, 16) == fld(H, 16)                  # same opaque atom from two builds
        @test fld(H, 16) != fld(H, 8)                   # different divisor -> different atom
        @test fld(H, 16) != fld(W, 16)                  # different numerator
        @test fld(fld(H, 2), 2) != fld(H, 4)            # conservative: no nested-fld algebra
        @test fld(H, W) isa JOLT.Poly                   # symbolic divisor -> opaque
        @test fld(100, B) isa JOLT.Poly                 # Int numerator, symbolic divisor
        @test mod(H, W) isa JOLT.Poly
        @test mod(100, B) isa JOLT.Poly

        # ÷ and % aliases (dims are ≥ 1, so truncating == flooring)
        @test H ÷ 16 == fld(H, 16)
        @test (16h) ÷ 16 == h
        @test (16h + 3) % 16 === 3

        @test_throws ErrorException fld(H, 0)           # non-positive divisors rejected
        @test_throws ErrorException fld(H, -2)
        @test_throws ErrorException mod(H, 0)
    end

    # -----------------------------------------------------------------
    # max / min: opaque at construction (facts-blind), commutative via
    # sorted args, resolved by canon when bounds prove an ordering.
    # -----------------------------------------------------------------
    @testset "max & min" begin
        @test max(H, H) == H                            # trivially equal
        @test max(H, W) == max(W, H)                    # commutative (sorted args)
        @test min(H, W) == min(W, H)
        @test max(H, W) != min(H, W)
        @test max(H, 2) isa JOLT.Poly

        F = JOLT.Facts()
        @test dims_equal(max(H, 1), H, F)               # default bound: dims ≥ 1
        @test dims_equal(min(H, 1), 1, F)
        @test !dims_equal(max(H, 2), H, F)              # H could be 1
        bound!(F, :H; lo = 2)
        @test dims_equal(max(H, 2), H, F)               # now provable

        @test realize(max(H, W), :H => 3, :W => 7) == 7
        @test realize(min(H - 2, 4), :H => 3) == 1
    end

    # -----------------------------------------------------------------
    # realize: plug in concrete sizes; opaque atoms re-apply their op.
    # -----------------------------------------------------------------
    @testset "realize" begin
        @test realize(5, Dict{Symbol,Int}()) === 5
        @test realize(H, :H => 7) == 7
        @test realize(2H + 3, :H => 10) == 23
        @test realize(H * W + 1, :H => 3, :W => 4) == 13
        @test realize(H^2, :H => 5) == 25
        @test realize(fld(H, 16), :H => 240) == 15
        @test realize(fld(H, 16) + 1, :H => 250) == 16  # floor really floors
        @test realize(mod(H, 16), :H => 250) == 10
        @test realize(fld(H, W), :H => 17, :W => 5) == 3   # opaque w/ symbolic divisor

        # a conv-shaped formula, written once, works both ways
        convout(x, p, k, s) = fld(x + 2p - k, s) + 1
        @test convout(28, 1, 3, 2) === 14               # static: plain Int arithmetic
        @test realize(convout(H, 1, 3, 2), :H => 28) == 14   # symbolic: same answer

        err = try realize(H + W, :H => 3); nothing catch e; sprint(showerror, e) end
        @test occursin("no value provided", err) && occursin("W", err)
    end

    # -----------------------------------------------------------------
    # Facts: equality between variables (union-find) — declared LATE, after
    # the dims already exist, and never rewriting the stored dims.
    # -----------------------------------------------------------------
    @testset "facts: variable equality & pins" begin
        F = JOLT.Facts()
        @test !dims_equal(B, D, F)                      # strict before declaration
        same_dim!(F, :B, :D)
        @test dims_equal(B, D, F)                       # provable after
        @test dims_equal(B + 1, D + 1, F)               # composes through arithmetic
        @test dims_equal(2B * W, 2D * W, F)
        @test B != D                                    # stored forms untouched (structural)

        # transitivity through the union-find
        same_dim!(F, :D, :E)
        @test dims_equal(B, todim(:E), F)

        # pin to a constant: canon resolves the var to the Int
        F2 = JOLT.Facts()
        pin!(F2, :D, 1)
        @test canon(D, F2) === 1
        @test dims_equal(D + 4, 5, F2)

        # contradictions are caught at declaration time
        F3 = JOLT.Facts()
        @test_throws ErrorException same_dim!(F3, todim(:X) + 1, todim(:X))   # X+1 == X
        @test_throws ErrorException same_dim!(F3, 3, 5)
        same_dim!(F3, 5, 5)                             # trivially true: a no-op
        same_dim!(F3, :X, :X)                           # self-equality: a no-op
        @test dims_equal(todim(:X), todim(:X), F3)

        # pinned vars merged by a later equality must agree
        F4 = JOLT.Facts()
        pin!(F4, :P, 3)
        pin!(F4, :Q, 5)
        @test_throws ErrorException same_dim!(F4, :P, :Q)   # 3 == 5 is unsatisfiable

        # facts migrate when a var with a rule is absorbed
        F5 = JOLT.Facts()
        pin!(F5, :P, 8)
        same_dim!(F5, :P, :Q)                           # P (pinned) merges into Q
        @test canon(todim(:P), F5) === 8
        @test canon(todim(:Q), F5) === 8
    end

    # -----------------------------------------------------------------
    # Facts: general polynomial equalities — oriented into rewrite rules
    # by solving for an isolated unit-coefficient atom (opaque atoms
    # preferred, matching JAX's floordiv(a,b) == c).
    # -----------------------------------------------------------------
    @testset "facts: polynomial rules" begin
        # N == B·T  (solves for N)
        F = JOLT.Facts()
        same_dim!(F, :N, B * T)
        @test dims_equal(todim(:N), B * T, F)
        @test dims_equal(todim(:N) + 1, B * T + 1, F)

        # H + W == 10  (solves one variable in terms of the other)
        F2 = JOLT.Facts()
        same_dim!(F2, H + W, 10)
        @test dims_equal(H, 10 - W, F2)
        @test dims_equal(H + W + 5, 15, F2)

        # 2H == 4W  (coefficient divides: H => 2W)
        F3 = JOLT.Facts()
        same_dim!(F3, 2H, 4W)
        @test dims_equal(H, 2W, F3)

        # 2H == 3W  (no unit atom, coefficients don't divide -> reject)
        F4 = JOLT.Facts()
        err = try same_dim!(F4, 2H, 3W); nothing catch e; sprint(showerror, e) end
        @test occursin("cannot orient", err)

        # ⌊H/16⌋ == h  (rule keyed on the OPAQUE atom — JAX's direction)
        F5 = JOLT.Facts()
        same_dim!(F5, fld(H, 16), :h)
        @test dims_equal(fld(H, 16), h, F5)
        @test dims_equal(fld(H, 16) * 16, 16h, F5)

        # ⌊H/16⌋·16 == H is CYCLIC as a rule (H appears inside the fld atom);
        # rejected with a pointer at the right tool
        F6 = JOLT.Facts()
        err = try same_dim!(F6, fld(H, 16) * 16, H); nothing catch e; sprint(showerror, e) end
        @test occursin("divisible!", err)

        # rule chains: x == y+1, then y == 2  =>  x canonicalizes to 3
        F7 = JOLT.Facts()
        same_dim!(F7, :x, todim(:y) + 1)
        same_dim!(F7, :y, 2)
        @test canon(todim(:x), F7) === 3
        @test dims_equal(todim(:x), 3, F7)

        # cycle guard: after x == y+1, declaring y == x-1 is redundant
        # (canonically 0 == 0) and must not create an infinite loop
        F8 = JOLT.Facts()
        same_dim!(F8, :x, todim(:y) + 1)
        same_dim!(F8, :y, todim(:x) - 1)                # provably identical: no-op
        @test dims_equal(todim(:x), todim(:y) + 1, F8)
        @test canon(todim(:x), F8) isa JOLT.Poly        # still symbolic, no hang
    end

    # -----------------------------------------------------------------
    # Facts: divisibility — declared on the ORIGINAL symbol, consulted at
    # comparison time; stored dims never rewritten.
    # -----------------------------------------------------------------
    @testset "facts: divisibility" begin
        F = JOLT.Facts()
        patch = fld(H, 16)                              # built BEFORE the declaration
        @test !dims_equal(patch * 16, H, F)             # strict: 225 would break it
        divisible!(F, :H, 16)
        @test dims_equal(patch * 16, H, F)              # now provable
        @test dims_equal(mod(H, 16), 0, F)
        @test dims_equal(fld(H + 32, 16), patch + 2, F)
        @test dims_equal(mod(H + 3, 16), 3, F)
        @test patch == fld(H, 16)                       # stored form still opaque

        @test JOLT.provably_divisible(H, 16, F)
        @test JOLT.provably_divisible(H, 8, F)          # 8 | 16 | H
        @test !JOLT.provably_divisible(H, 32, F)
        @test JOLT.provably_divisible(H + 32, 16, F)
        @test !JOLT.provably_divisible(H + 3, 16, F)
        @test JOLT.provably_divisible(3 * H, 48, F)

        # two declarations compose by lcm
        divisible!(F, :H, 6)
        @test JOLT.provably_divisible(H, 48, F)         # lcm(16,6)=48
        @test_throws ErrorException divisible!(F, :H, 1)   # k must be ≥ 2

        # concrete numbers short-circuit
        @test JOLT.provably_divisible(48, 16, F)
        @test !JOLT.provably_divisible(50, 16, F)
    end

    # -----------------------------------------------------------------
    # ModularSet analyzer: divisibility of DERIVED dims is proven bottom-up
    # (sums take gcd, products multiply moduli) — no rewriting involved.
    # -----------------------------------------------------------------
    @testset "analyzer: modular sets" begin
        F = JOLT.Facts()
        divisible!(F, :A, 4)
        divisible!(F, :C, 4)
        A, C = todim(:A), todim(:C)
        @test JOLT.provably_divisible(A + C, 4, F)      # gcd(4,4)
        @test !JOLT.provably_divisible(A + C, 8, F)     # 4+4=8 but also 4+8=12
        @test JOLT.provably_divisible(A * C, 16, F)     # moduli multiply
        @test JOLT.provably_divisible(2A, 8, F)
        @test JOLT.provably_divisible(A * A, 16, F)     # powers too
        @test !JOLT.provably_divisible(A + 2, 4, F)     # residue 2
        @test JOLT.provably_divisible(A + 8, 4, F)      # residue 0
        @test JOLT.provably_divisible(16 * B, 16, F)    # structural coefficient
        @test !JOLT.provably_divisible(B, 2, F)         # nothing known about B
    end

    # -----------------------------------------------------------------
    # ConstIntBound analyzer: interval propagation with the JAX default
    # (every dim ≥ 1), tightened by bound!, saturating (no overflow).
    # -----------------------------------------------------------------
    @testset "analyzer: bounds" begin
        F = JOLT.Facts()
        @test JOLT.provably_ge(H, 1, F)                 # the default: dims ≥ 1
        @test JOLT.provably_ge(H + W, 2, F)
        @test JOLT.provably_ge(H * W, 1, F)
        @test !JOLT.provably_ge(H, 2, F)                # H could be 1
        @test JOLT.provably_ge(H, H, F)                 # reflexive (difference is 0)
        @test JOLT.provably_ge(2H, H, F)                # 2H - H = H ≥ 1 ≥ 0

        bound!(F, :H; lo = 2, hi = 10)
        @test JOLT.provably_ge(H, 2, F)
        @test JOLT.provably_ge(10, H, F)
        @test JOLT.provably_ge(2H + 3, 7, F)            # [7, 23]
        @test !JOLT.provably_ge(2H + 3, 8, F)           # lo is 7
        @test JOLT.provably_ge(H * W, 2, F)             # [2,10]×[1,∞)

        # bounds tighten monotonically; contradictions rejected
        bound!(F, :H; lo = 3)
        @test JOLT.provably_ge(H, 3, F)
        @test_throws ErrorException bound!(F, :H; lo = 11)   # lo > current hi=10

        # opaque bounds: mod(·,k) ∈ [0,k-1]; fld propagates through
        @test JOLT.provably_ge(15, mod(B, 16), F)
        @test JOLT.provably_ge(fld(B, 4), 0, F)
        bound!(F, :B; lo = 100)
        @test JOLT.provably_ge(fld(B, 4), 25, F)

        # saturation: unbounded products don't overflow, remain comparable
        huge = todim(:P) * todim(:Q) * typemax(Int32)
        @test JOLT.provably_ge(huge, 1, F)
        @test !JOLT.provably_ge(1, huge, F)
    end

    # -----------------------------------------------------------------
    # Interplay: a var carrying facts absorbed by a later equality.
    # -----------------------------------------------------------------
    @testset "facts: merge interactions" begin
        F = JOLT.Facts()
        divisible!(F, :B, 16)
        bound!(F, :B; lo = 32)
        same_dim!(F, :B, :D)                            # B merges into D
        @test JOLT.provably_divisible(B, 16, F)         # divisibility survives the merge
        @test JOLT.provably_divisible(D, 16, F)         # and transfers to the rep
        @test JOLT.provably_ge(B, 32, F)                # bounds survive too
        @test dims_equal(fld(B, 16) * 16, D, F)         # everything composes
    end

    # -----------------------------------------------------------------
    # Strict broadcast / contract: prove or reject; overrides feed the
    # prover rather than bypassing it. TF/JAX left-padding.
    # -----------------------------------------------------------------
    @testset "broadcast & contract" begin
        F = JOLT.Facts()
        @test JOLT.broadcast_dim(1, 10, F) === 10       # a literal 1 stretches
        @test JOLT.broadcast_dim(10, 1, F) === 10
        @test JOLT.broadcast_dim(10, 10, F) === 10
        @test JOLT.broadcast_dim(B, 1, F) == B
        @test JOLT.broadcast_dim(1, B, F) == B
        @test JOLT.broadcast_dim(B, B, F) == B
        @test_throws ErrorException JOLT.broadcast_dim(10, 20, F)
        @test_throws ErrorException JOLT.broadcast_dim(B, 10, F)   # never assumed equal
        @test_throws ErrorException JOLT.broadcast_dim(B, D, F)    # nor provably-equal
        @test_throws ErrorException JOLT.broadcast_dim(B, W, F)

        # (:A,:B,:C) .* (:A,:D,:C) — the canonical reject; then each override
        F1 = JOLT.Facts()
        sa = (todim(:A), B, todim(:C))
        sb = (todim(:A), D, todim(:C))
        @test_throws ErrorException JOLT.broadcast_shapes(sa, sb, F1)
        same_dim!(F1, :B, :D)                           # override 1: declare equal
        @test JOLT.broadcast_shapes(sa, sb, F1) == sa
        F2 = JOLT.Facts()
        pin!(F2, :D, 1)                                 # override 2: declare one is 1
        @test JOLT.broadcast_shapes(sa, sb, F2) == sa   # D stretches away

        # left-padding: (N, M, 10) ⊕ (10,) -> (N, M, 10)
        F3 = JOLT.Facts()
        @test JOLT.broadcast_shapes((todim(:N), todim(:M), 10), (10,), F3) ==
              (todim(:N), todim(:M), 10)
        @test JOLT.broadcast_shapes((10,), (todim(:N), todim(:M), 10), F3) ==
              (todim(:N), todim(:M), 10)                # symmetric
        @test JOLT.broadcast_shapes((), (3, 4), F3) == (3, 4)    # scalar vs matrix
        @test JOLT.broadcast_shapes((), (), F3) == ()            # scalar vs scalar
        @test JOLT.broadcast_shapes((B, 1), (1, W), F3) == (B, W)  # both stretch

        # contraction (matmul inner dim): equal or reject
        @test JOLT.contract_dim(4, 4, F3) === nothing
        @test JOLT.contract_dim(B, B, F3) === nothing
        @test_throws ErrorException JOLT.contract_dim(4, 5, F3)
        @test_throws ErrorException JOLT.contract_dim(B, D, F3)
        same_dim!(F3, :B, :D)
        @test JOLT.contract_dim(B, D, F3) === nothing

        # derived dims contract when provably equal
        F4 = JOLT.Facts()
        @test JOLT.contract_dim(fld(H, 2), fld(H, 2), F4) === nothing
        @test_throws ErrorException JOLT.contract_dim(fld(H, 2), fld(W, 2), F4)
    end

    # -----------------------------------------------------------------
    # check_facts: the run-time gate — every declaration re-verified
    # against concrete sizes before data would reach IREE.
    # -----------------------------------------------------------------
    @testset "check_facts (runtime gate)" begin
        F = JOLT.Facts()
        same_dim!(F, :B, :D)
        same_dim!(F, :N, B * T)
        divisible!(F, :H, 16)
        bound!(F, :T; lo = 2, hi = 512)

        good = Dict(:B => 8, :D => 8, :N => 64, :T => 8, :H => 224)
        @test check_facts(good, F) === nothing

        for (env, needle) in [
            (Dict(:B => 8, :D => 9, :N => 72, :T => 9, :H => 224), "B == D"),      # eq broken
            (Dict(:B => 8, :D => 8, :N => 65, :T => 8, :H => 224), "vs"),          # N ≠ B·T
            (Dict(:B => 8, :D => 8, :N => 64, :T => 8, :H => 225), "16"),          # 16 ∤ 225
            (Dict(:B => 8, :D => 8, :N => 8,  :T => 1, :H => 224), "≤"),           # T below lo
        ]
            err = try check_facts(env, F); nothing catch e; sprint(showerror, e) end
            @test err !== nothing && occursin("violated", err) && occursin(needle, err)
        end

        # a value missing from env is its own clear error
        err = try check_facts(Dict(:B => 8), F); nothing catch e; sprint(showerror, e) end
        @test occursin("no value provided", err)
    end

    # -----------------------------------------------------------------
    # MLIR lowering + the StableHLO secondary check (context-free calls).
    # -----------------------------------------------------------------
    @testset "MLIR interop" begin
        @test JOLT.dim_to_mlir(4) === 4
        dyn = JOLT.dim_to_mlir(B)
        @test JOLT.IR.isdynsize(dyn)
        @test JOLT.dim_to_mlir(fld(H, 16)) == dyn       # every Poly lowers to ?

        @test JOLT.mlir_ok(4, 4)                        # concrete agree
        @test !JOLT.mlir_ok(4, 5)                       # concrete disagree -> rule bug
        @test JOLT.mlir_ok(4, dyn)                      # MLIR lost the info: fine
        @test JOLT.mlir_ok(B, dyn)                      # symbolic vs ?: fine
        @test JOLT.mlir_ok(B, 7)                        # symbolic vs concrete: fine
    end

    # -----------------------------------------------------------------
    # Display: exact for the simple forms (Tensor show depends on them),
    # loose for the composites.
    # -----------------------------------------------------------------
    @testset "display" begin
        @test string(B) == "B"
        @test string(2H) == "2H"
        @test string(2H + 3) == "2H + 3"
        @test string(H - 5) == "H - 5"
        @test string(-H) == "-H"
        @test string(H^2) == "H^2"
        s = string(H * W + H - W - 1)
        @test occursin("H·W", s) && occursin("- 1", s)
        @test string(fld(H, 16)) == "⌊H/16⌋"
        @test string(fld(H - 1, 2)) == "⌊(H - 1)/2⌋"    # multi-term args get parens
        @test string(mod(H, 16)) == "(H mod 16)"
        @test occursin("max", string(max(H, W)))
    end

    # -----------------------------------------------------------------
    # No dispatch ambiguities were introduced by the operator overloads.
    # -----------------------------------------------------------------
    @testset "dispatch hygiene" begin
        @test isempty(Test.detect_ambiguities(JOLT))

        # loosely-typed envs convert instead of MethodError-ing
        @test realize(5, Dict()) === 5
        @test realize(H, Dict{Any,Any}(:H => 7)) == 7

        # unsupported-by-design operations get curated errors, not MethodErrors
        @test_throws ErrorException 2^H                 # symbolic exponent
        @test_throws ErrorException H^W
        @test_throws ErrorException H < W               # no build-time order
        @test_throws ErrorException H < 5
        @test_throws ErrorException 5 > H

        # structurally-equal fld/mod args reduce (dims are ≥ 1)
        @test fld(H, H) === 1
        @test mod(H, H) === 0
        @test fld(H + 1, H + 1) === 1

        # fld/mod by 1 are identities
        @test fld(H, 1) == H
        @test mod(H, 1) === 0

        # realize through nested opaque atoms
        nested = fld(max(H, W) + mod(H, 3), 2)
        @test realize(nested, :H => 7, :W => 5) == fld(7 + 1, 2)
    end

    # -----------------------------------------------------------------
    # Regressions from the adversarial verification run. Each block pins a
    # confirmed defect: the ±_INF sentinel leaking into bound arithmetic,
    # rule migration during variable merges, and declaration-log pollution
    # on rejected constraints.
    # -----------------------------------------------------------------
    @testset "regression: ∞ sentinel is not a number" begin
        F = JOLT.Facts()
        huge = 2 * JOLT._INF
        # ∞ + (-∞) must widen conservatively, never cancel to 0
        @test !JOLT.provably_ge(huge * W, H, F)         # H can exceed any finite claim
        @test !dims_equal(max(huge * W, H), huge * W, F)  # so max cannot resolve
        # ⌊∞/k⌋ must stay ∞, not become a finite bound
        F2 = JOLT.Facts()
        divisible!(F2, :H, 16)
        K = fld(JOLT._INF, 16) + 1
        @test !JOLT.provably_ge(K, fld(H, 16), F2)      # quotient is unbounded above
        # sanity: ordinary magnitudes still prove fine
        @test JOLT.provably_ge(H + 1, 2, F)
    end

    @testset "regression: rule migration through merges" begin
        # satisfiable facts must not crash canon (cyclic migrated rule):
        # a == 2⌊b/2⌋+1 ("a is odd-ish") then a == b is representable only by a
        # rule b => …b…, which the occurs check must refuse — as a loud error,
        # with the store rolled back, not a StackOverflow later.
        F = JOLT.Facts()
        same_dim!(F, :a, 2 * fld(todim(:b), 2) + 1)
        @test_throws ErrorException same_dim!(F, :a, :b)
        @test canon(todim(:b), F) == todim(:b)          # no crash, store consistent
        @test dims_equal(todim(:a), 2 * fld(todim(:b), 2) + 1, F)  # first fact intact

        # unsatisfiable facts must error loudly at declaration, not silently
        # install b => b+1
        F2 = JOLT.Facts()
        same_dim!(F2, :a, todim(:b) + 1)
        @test_throws ErrorException same_dim!(F2, :a, :b)   # a==b+1 ∧ a==b is absurd
        @test dims_equal(todim(:a), todim(:b) + 1, F2)      # declared fact survives
        @test canon(todim(:b), F2) == todim(:b)             # no runaway rewriting
    end

    @testset "regression: rejected constraints leave no trace" begin
        # unorientable: rejected, so the runtime gate must NOT enforce it
        F = JOLT.Facts()
        @test_throws ErrorException same_dim!(F, 2 * todim(:a), 3 * todim(:b))
        @test isempty(F.decls)
        @test check_facts(Dict(:a => 3, :b => 1), F) === nothing

        # contradiction against an accepted pin: the pin survives, the
        # rejected equality is not logged, and its env still passes
        F2 = JOLT.Facts()
        pin!(F2, :a, 3)
        @test_throws ErrorException same_dim!(F2, :a, 5)
        @test check_facts(Dict(:a => 3), F2) === nothing
        @test_throws ErrorException check_facts(Dict(:a => 4), F2)  # pin still enforced

        # merge-contradiction (two pins) rolls back the union entirely
        F3 = JOLT.Facts()
        pin!(F3, :B, 3)
        pin!(F3, :D, 5)
        @test_throws ErrorException same_dim!(F3, :B, :D)
        @test canon(B, F3) === 3                        # both pins intact,
        @test canon(D, F3) === 5                        # no half-merged state
        @test !dims_equal(B, D, F3)
        @test check_facts(Dict(:B => 3, :D => 5), F3) === nothing
    end

    # -----------------------------------------------------------------
    # Gap coverage from the test critic: fact kinds interacting, facts
    # reaching inside opaque args, hi-driven resolution, absorbed names,
    # and the canon-is-comparison-only boundary.
    # -----------------------------------------------------------------
    @testset "facts: interactions & reach" begin
        # pin! on a Poly expression (not just a Symbol)
        F = JOLT.Facts()
        pin!(F, H + W, 10)                              # orients to a rule
        @test dims_equal(H, 10 - W, F)
        @test_throws ErrorException check_facts(Dict(:H => 4, :W => 4), F)
        @test check_facts(Dict(:H => 4, :W => 6), F) === nothing

        # divisibility + pin on the SAME variable: consistent pair proves both
        F2 = JOLT.Facts()
        divisible!(F2, :H, 16)
        pin!(F2, :H, 32)                                # 16 | 32 ✓ (solves the quotient)
        @test canon(H, F2) === 32
        @test dims_equal(fld(H, 16), 2, F2)
        # ...and an inconsistent pair is rejected at declaration
        F3 = JOLT.Facts()
        divisible!(F3, :H, 16)
        @test_throws ErrorException pin!(F3, :H, 33)    # 16 ∤ 33: unorientable

        # facts reach INSIDE opaque atom arguments (catom re-canonicalizes args)
        F4 = JOLT.Facts()
        pin!(F4, :H, 12)
        @test dims_equal(fld(H, 4), 3, F4)              # ⌊12/4⌋ folds
        @test dims_equal(mod(H, 5), 2, F4)
        @test dims_equal(max(H, W), 12, F4) == false    # W could exceed 12...
        bound!(F4, :W; hi = 12)
        @test dims_equal(max(H, W), 12, F4)             # ...until its hi says otherwise

        # bound! with only hi; hi-driven min resolution
        F5 = JOLT.Facts()
        bound!(F5, :T; hi = 8)
        @test JOLT.provably_ge(8, T, F5)
        @test dims_equal(min(T, 8), T, F5)
        @test dims_equal(max(T, 8), 8, F5)

        # declarations through an ABSORBED union-find name still bite
        F6 = JOLT.Facts()
        same_dim!(F6, :B, :D)                           # B absorbed into D
        divisible!(F6, :B, 4)                           # declared via the old name
        bound!(F6, :B; lo = 8)
        @test JOLT.provably_divisible(D, 4, F6)         # lands on the representative
        @test JOLT.provably_ge(D, 8, F6)
        @test JOLT.provably_divisible(B, 4, F6)

        # contract provable only THROUGH facts (divisibility route)
        F7 = JOLT.Facts()
        divisible!(F7, :H, 16)
        @test JOLT.contract_dim(fld(H, 16) * 16, H, F7) === nothing
        @test JOLT.broadcast_dim(fld(H, 16) * 16, H, F7) == fld(H, 16) * 16

        # canon output is for COMPARISON only: a minted quotient variable is
        # not a user symbol and cannot be realized
        F8 = JOLT.Facts()
        divisible!(F8, :H, 16)
        c = canon(H, F8)
        @test c isa JOLT.Poly && c != H                 # rewritten to 16·(H÷16)
        @test_throws ErrorException realize(c, :H => 32)   # minted var has no env entry
        @test realize(H, :H => 32) == 32                # the STORED dim realizes fine
    end

end
