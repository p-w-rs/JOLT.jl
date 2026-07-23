# =====================================================================
# Einsum — general Einstein-summation contraction, and its friendly faces
# `matmul` (⊡, matmul.jl) and `dot` (⋅).
#
# einsum is NOT a new taped primitive and needs NO new StableHLO op. It is a
# GRAPH BUILDER: it parses the equation and lowers it to a composition of ops
# that already exist — reduce_sum (sum an index away), permutedims (reorder
# axes), reshape (group axes), and matmul (the one contraction). Because every
# piece is a taped op with its own vjp, einsum's gradient falls out for free —
# and since that gradient is again built from the same ops (each closed under
# vjp), it differentiates to ANY order with no einsum-specific backward code.
#
# The three phases (NumPy/opt_einsum's shape, specialised to left-to-right):
#   1. reduce  — an index that occurs in exactly ONE operand and not in the
#                output is summed away first (reduce_sum).
#   2. contract — fold operands left-to-right; each pair is aligned to
#                [batch…, free, contract] via permutedims+reshape and handed to
#                `matmul`, which is the sole dot_general.
#   3. finish  — sum any leftover non-output index, then permute to the
#                requested output order.
#
# NOT supported (rejected at build with a message, never silently mis-lowered):
#   * a repeated index WITHIN one operand ("ii->i" diagonal, "ii->" trace) — its
#     gradient is a scatter onto a diagonal, which is not an einsum; extract the
#     diagonal explicitly instead.
#   * ellipsis ("...") — write the batch axes out.
# =====================================================================

# --- parse "ij,jk->ik" -> ([['i','j'],['j','k']], ['i','k']) --------
function _parse_einsum(spec::AbstractString)
    occursin("...", spec) &&
        error("einsum: ellipsis (`...`) is not supported yet — write the batch axes explicitly ($(repr(spec)))")
    parts = split(spec, "->")
    length(parts) == 2 ||
        error("einsum: the spec must contain exactly one `->` (got $(repr(spec))); e.g. \"ij,jk->ik\"")
    ins = [collect(strip(g)) for g in split(strip(parts[1]), ",")]
    out = collect(strip(parts[2]))
    for c in Iterators.flatten((Iterators.flatten(ins), out))
        isletter(c) || error("einsum: index labels must be letters; got $(repr(c)) in $(repr(spec))")
    end
    return ins, out
end

_labelcounts(labels) = (c = Dict{Char,Int}(); for g in labels, ch in g; c[ch] = get(c, ch, 0) + 1; end; c)

# no-op guards so identity permutes/reshapes don't litter the tape
_maybe_permute(x, perm) = perm == collect(1:length(perm)) ? x : permutedims(x, perm)
_maybe_reshape(x, dims) = Tuple(dims) == size(x) ? x : reshape(x, dims...)

# Reorder X so its axes are [g1…, g2…, g3…] (g1/g2/g3 partition LX), then
# COLLAPSE g2 into one axis and g3 into one axis (g1 stays as separate axes):
#     result shape = (g1 dims…, prod(g2 dims), prod(g3 dims))
# Used to bring a contraction operand into matmul's (batch…, m, k) layout.
function _align_collapse(X, LX, g1, g2, g3)
    order = vcat(g1, g2, g3)
    perm  = [findfirst(==(c), LX) for c in order]
    Xp    = _maybe_permute(X, perm)
    sz    = size(Xp); n1 = length(g1); n2 = length(g2)
    g1dims = sz[1:n1]
    g2dims = sz[n1+1:n1+n2]
    g3dims = sz[n1+n2+1:end]
    m = isempty(g2dims) ? 1 : reduce(*, g2dims)
    k = isempty(g3dims) ? 1 : reduce(*, g3dims)
    return _maybe_reshape(Xp, (g1dims..., m, k))
end

# Contract two labeled tensors, keeping only the labels in `keep` (plus every
# operand-unique label, which is carried through). Shared labels split into
# BATCH (kept) and CONTRACT (summed); operand-unique labels are FREE.
function _contract2(A, LA, B, LB, keep)
    setA, setB = Set(LA), Set(LB)
    shared   = [c for c in LA if c in setB]
    batch    = [c for c in shared if c in keep]
    contract = [c for c in shared if !(c in keep)]
    Afree    = [c for c in LA if !(c in setB)]
    Bfree    = [c for c in LB if !(c in setA)]
    A2 = _align_collapse(A, LA, batch, Afree, contract)   # (batch…, M, K)
    B2 = _align_collapse(B, LB, batch, contract, Bfree)   # (batch…, K, N)
    C  = matmul(A2, B2)                                    # (batch…, M, N)
    dimof(X, LX, c) = size(X)[findfirst(==(c), LX)]
    target = (size(C)[1:length(batch)]...,
              (dimof(A, LA, c) for c in Afree)...,
              (dimof(B, LB, c) for c in Bfree)...)
    return _maybe_reshape(C, target), vcat(batch, Afree, Bfree)
end

"""
    einsum("ij,jk->ik", A, B)   (also einsum of 1 or n operands)

General Einstein summation. Labels are single letters; an index in the output is
kept, one shared across operands but absent from the output is contracted (summed),
and one unique to a single operand and absent from the output is summed away.
Examples: `"ik,kj->ij"` matmul, `"bik,bkj->bij"` batched matmul, `"i,i->"` dot,
`"i,j->ij"` outer, `"ij->ji"` transpose, `"ij->i"` row-sum.

Built from `reduce_sum`/`permutedims`/`reshape`/`matmul`, so it is differentiable
to any order with no special-casing. Symbolic dims flow through (shared indices
must be provably equal — declare `same_dim!` otherwise).
"""
function einsum(spec::AbstractString, ops::AbstractTensor...)
    ins, out = _parse_einsum(spec)
    length(ins) == length(ops) ||
        error("einsum: $(repr(spec)) names $(length(ins)) operand group(s) but $(length(ops)) tensor(s) were passed")
    for (k, (grp, t)) in enumerate(zip(ins, ops))
        length(grp) == ndims(t) ||
            error("einsum: operand $k is rank $(ndims(t)) but its group \"$(String(grp))\" names $(length(grp)) axis/axes ($(repr(spec)))")
        allunique(grp) ||
            error("einsum: a repeated index within one operand (\"$(String(grp))\") is not supported — " *
                  "diagonal/trace has no einsum gradient. Extract the diagonal explicitly first. ($(repr(spec)))")
    end
    allunique(out) || error("einsum: repeated output index in \"$(String(out))\" ($(repr(spec)))")

    # label -> Dim, checking every shared occurrence is PROVABLY equal (strict, like every op)
    dimof = Dict{Char,Dim}()
    for (grp, t) in zip(ins, ops), (c, d) in zip(grp, size(t))
        if haskey(dimof, c)
            dims_equal(dimof[c], d) ||
                error("einsum: index $(repr(c)) has sizes $(dimof[c]) vs $d that are not provably equal; " *
                      "declare same_dim! if they are ($(repr(spec)))")
        else
            dimof[c] = d
        end
    end
    for c in out
        haskey(dimof, c) ||
            error("einsum: output index $(repr(c)) appears in no operand, so its size is undetermined ($(repr(spec)))")
    end

    labels = [copy(g) for g in ins]                 # working copies (mutate as we reduce/fold)
    tens   = collect(AbstractTensor, ops)

    # Phase 1 — sum indices unique to one operand and not in the output.
    cnt = _labelcounts(labels)
    for k in eachindex(tens)
        drop = [c for c in labels[k] if cnt[c] == 1 && !(c in out)]
        isempty(drop) && continue
        tens[k]   = reduce_sum(tens[k], sort!([findfirst(==(c), labels[k]) for c in drop]))
        labels[k] = [c for c in labels[k] if !(c in drop)]
    end

    # Phase 2 — fold operands left-to-right.
    curL, curT = labels[1], tens[1]
    for k in 2:length(tens)
        later = Set{Char}()
        for j in (k+1):length(tens); union!(later, labels[j]); end
        curT, curL = _contract2(curT, curL, tens[k], labels[k], union(Set(out), later))
    end

    # Phase 3 — sum any leftover non-output index (single-operand einsums), then
    # permute to the requested output order.
    leftover = [c for c in curL if !(c in out)]
    if !isempty(leftover)
        curT = reduce_sum(curT, sort!([findfirst(==(c), curL) for c in leftover]))
        curL = [c for c in curL if c in out]
    end
    curL == out && return curT
    return permutedims(curT, [findfirst(==(c), curL) for c in out])
end

# --- dot / ⋅ : the full inner product  Σ aᵢbᵢ  (Frobenius for higher rank) ---
# `⋅` matches Julia's LinearAlgebra.⋅ = dot and the mathematical convention.
# (Elementwise/Hadamard is `.*`/`⊙`, NOT this.)
function dot(a::AbstractTensor, b::AbstractTensor)
    ndims(a) == ndims(b) ||
        error("dot: operands must have equal rank (got $(ndims(a)) and $(ndims(b))); dot is the full inner product Σ aᵢbᵢ")
    ndims(a) <= 26 || error("dot: rank $(ndims(a)) exceeds the 26 single-letter labels; use einsum directly")
    labs = String(['a' + (i - 1) for i in 1:ndims(a)])
    return einsum("$labs,$labs->", a, b)
end
const ⋅ = dot     # U+22C5 (\cdot)
