# =====================================================================
# compile & export — the user-facing surface.
#
#     fn, vars = compile(rng, inputs, outputs; backend = IREE_CPU)
#     results  = fn(vars, argvalues...)
#
# `vars` is a ComponentArray of every :variables tensor (nested by scope), its
# init values realized reproducibly from `rng`. `fn` binds `vars` and the call's
# argument values into the func's input order and runs the backend.
#
#     text = export_stablehlo(inputs, outputs)   # same graph, no backend
# =====================================================================

# ---- reproducible per-variable seeding -------------------------------------
# Base.hash is process-salted and default RNG streams drift across Julia
# versions, so derive a version-stable UInt64 seed with SHA-256 over a length-
# prefixed (base, path) and feed a StableRNG (a version-stable stream).
_seed_base(rng::Integer)     = UInt64(rng)
_seed_base(rng::AbstractRNG) = rand(rng, UInt64)

function _canonical(base::Integer, path::Tuple)
    io = IOBuffer(); write(io, "JOLT/v1")
    b = string(base); write(io, hton(UInt32(sizeof(b)))); write(io, b)
    write(io, hton(UInt32(length(path))))
    for sym in path
        n = String(sym); write(io, hton(UInt32(sizeof(n)))); write(io, n)
    end
    return take!(io)
end
variable_rng(base::Integer, path::Tuple) =
    StableRNG(foldl((a, b) -> (a << 8) | UInt64(b), SHA.sha256(_canonical(base, path))[1:8]; init = UInt64(0)))

# ---- vars ComponentArray (nested by scope; the :variables head is dropped) --
function _nest(pairs::Vector)
    leaves = Pair{Symbol,Any}[]; subs = Dict{Symbol,Vector}()
    for (path, v) in pairs
        length(path) == 1 ? push!(leaves, path[1] => v) :
                            push!(get!(subs, path[1], Pair[]), path[2:end] => v)
    end
    return merge((; leaves...), (; (k => _nest(v) for (k, v) in subs)...))
end

function _materialize_vars(s::Session, base::Integer)
    pairs = Pair{Any,Any}[]
    for (path, t) in s.names
        t isa Variable || continue
        arr = initializer(t)(variable_rng(base, path), eltype(t), size(t)...)
        push!(pairs, path[2:end] => arr)              # drop the :variables head
    end
    return _rehost(ComponentArray(_nest(pairs)))
end

# Re-back a ComponentArray with a page-aligned, anonymously-mmap'd arena, keeping
# its axes (so `vars` stays a normal DENSE ComponentArray that Optimisers mutates
# in place). Zero-copy IREE import needs the buffer's base pointer 64-byte aligned
# (IREE_HAL_HEAP_BUFFER_ALIGNMENT), which a plain Julia Vector does not guarantee;
# a page boundary (16 KB on Apple silicon) satisfies that AND is the vm_allocate/
# mmap-backed pointer Metal's newBufferWithBytesNoCopy requires — one arena, both
# backends. The arena must outlive the compiled `fn`: IREE imports it NON-OWNING,
# so it (via `vars`) is kept live for the session's lifetime; mutate it IN PLACE
# (Optimisers.update!) — reallocating `vars` would leave IREE aliasing stale bytes.
function _rehost(ca::ComponentArray)
    d = getdata(ca)
    isempty(d) && return ca                             # a 0-length mmap is invalid
    arena = Mmap.mmap(Vector{eltype(d)}, length(d))     # anonymous ⇒ page-aligned
    copyto!(arena, d)
    return ComponentArray(arena, getaxes(ca)...)
end

# ====================================================================
# The `vars` arena as an in/out call argument (zero-copy write-back).
# Variables are NOT compiled in as individual inputs and NOT served by a
# parameter provider; the whole dense `vars` arena rides as ONE flat call
# argument `%__arena`, out of which the graph slices each Variable. When any
# variable is `assign!`'d, `%__arena` carries `{iree.abi.output = 0}` and the
# graph `dynamic_update_slice`s the new value(s) back into it — so IREE aliases
# result 0 onto the arena and mutates Julia's bytes IN PLACE (see shim.c). With
# no assign!, the arena is a read-only input (zero-copy read, as before).
# ====================================================================

# MLIR spellings, matching module.jl's mlir_type (REVERSE-DIMS: Julia (m,n) → nxm).
_eltstr(::Type{Float32}) = "f32"; _eltstr(::Type{Float64}) = "f64"; _eltstr(::Type{Float16}) = "f16"
_eltstr(::Type{Int32}) = "i32"; _eltstr(::Type{Int64}) = "i64"; _eltstr(::Type{Int8}) = "i8"; _eltstr(::Type{Bool}) = "i1"
_dimstr(d) = d isa Integer ? string(d) : "?"        # symbolic/dynamic dim → MLIR `?` (cf. dim_to_mlir)
_tystr(::Type{T}, shape) where {T} =
    isempty(shape) ? "tensor<$(_eltstr(T))>" :
    "tensor<$(join((_dimstr(d) for d in reverse(shape)), 'x'))x$(_eltstr(T))>"

struct CompiledFn
    exe::IREEExecutable
    argplan::Vector{Int}      # @main Argument #k (after the arena) ← value of call arg #argplan[k]
    nout::Int                 # number of USER outputs returned to the caller
    state_paths::Vector{Any}  # ComponentArray path of each `assign!`ed variable, in output order
    repack::Vector{Any}       # per user output: ComponentArray axes to rebuild a PackedGrad, else nothing
    arena_len::Int            # element length of the compiled `vars` arena — layout guard
    arena_elt::DataType       # element type of that arena
    arena_axes::Any           # ComponentArray axes of the compiled `vars` — layout guard
end

# The compiled @main takes the `vars` arena as input 0, then the Arguments. For
# each Argument in graph/block order, the index into the user's `inputs`.
function _arg_plan(s::Session, inputs::Vector{<:AbstractTensor})
    argidx = IdDict(t => i for (i, t) in enumerate(inputs))
    plan = Int[]
    for t in s.argvars
        t isa Argument || continue
        haskey(argidx, t) || error("compile: an Argument feeds the graph but isn't in `inputs`")
        push!(plan, argidx[t])
    end
    return plan
end

# Element (offset, length) of each Variable within `vars`' contiguous arena, keyed
# by CA path (registry path minus the :variables head). Read by ADDRESS from the
# realized ComponentArray, so it is exactly the shared buffer's packing.
function _var_layout(s::Session, vars::ComponentArray)
    base = UInt(pointer(getdata(vars))); T = eltype(vars)
    layout = Dict{Tuple,Tuple{Int,Int}}()
    for (path, t) in s.names
        t isa Variable || continue
        capath = path[2:end]
        sub = foldl(getproperty, capath; init = vars)
        layout[capath] = (Int((UInt(pointer(sub)) - base) ÷ sizeof(T)), length(sub))
    end
    return layout
end

# Assemble the module text: the graph as `private @__jolt_graph` (returning the
# user outputs THEN each assigned variable's new value), plus a `@main` wrapper
# that takes the flat `vars` arena as READ-ONLY %__arena (call input 0), slices
# each Variable out of it with `stablehlo.slice`, calls the graph, and returns
# every graph result unchanged. The arena is never written inside the graph — the
# `assign!` updates come back as the trailing results, and `fn` copies them into
# the arena's slots after the call (see the CompiledFn call). Reads are zero-copy
# on unified memory (a Julia-side mutation between calls is seen live); the write
# is a small host-side copy of only the changed variables.
#
# (In-place device-side write-back — aliasing the arena via `iree.abi.output` and
# `flow.tensor.update` — was prototyped but is UNSAFE on iree-compile 3.12: that
# path lowers to the `hal.tensor.alias` hint, which side-steps alias analysis, so
# a read of an OLD value racing the in-place write is miscompiled order-dependently
# and the protective `flow.tensor.slice` clone gets elided. Revisit if IREE gains
# liveness-safe donation.)
#
# `raw` are the user outputs (in order); `targets` are the assigned Variables (in
# the order their new values trail the graph outputs). Reverse-dims is a no-op for
# the 1-D arena; per-Variable slices reshape to each Variable's reverse-dims type.
function _arena_text(s::Session, graph_mod::IR.Module, raw::Vector{<:AbstractTensor},
                     targets::Vector{<:AbstractTensor}, vars::ComponentArray,
                     layout::Dict{Tuple,Tuple{Int,Int}})
    T = eltype(vars); elt = _eltstr(T)
    tot = length(getdata(vars)); flatty = "tensor<$(tot)x$(elt)>"
    path_of = IdDict(t => p for (p, t) in s.names)

    gtxt = stablehlo_text(graph_mod)                       # module { func.func @__jolt_graph ... }
    i = findfirst('{', gtxt); j = findlast('}', gtxt)      # strip the outer module wrapper
    graph_func = strip(gtxt[nextind(gtxt, i):prevind(gtxt, j)])
    graph_func = replace(graph_func, "func.func @__jolt_graph" =>
                                     "func.func private @__jolt_graph"; count = 1)

    args   = [t for t in s.argvars if t isa Argument]
    argno  = IdDict(t => k - 1 for (k, t) in enumerate(args))
    argsig = join(("%a$(argno[t]): $(_tystr(eltype(t), size(t)))" for t in args), ", ")
    fullsig = isempty(args) ? "%__arena: $flatty" : "%__arena: $flatty, $argsig"

    # @__jolt_graph returns the user outputs then each assigned variable's new value;
    # @main returns exactly those, unchanged.
    res_tys = vcat([_tystr(eltype(o), size(o)) for o in raw],
                   [_tystr(eltype(v), size(v)) for v in targets])
    nres    = length(res_tys)
    outsig  = nres == 1 ? res_tys[1] : "(" * join(res_tys, ", ") * ")"

    body = IOBuffer()
    # slice each Variable out of the flat arena (read-only), in @__jolt_graph input order.
    callparams = String[]; callptys = String[]; varno = 0
    for t in s.argvars
        push!(callptys, _tystr(eltype(t), size(t)))
        if t isa Argument
            push!(callparams, "%a$(argno[t])")
        else
            off, len = layout[path_of[t][2:end]]
            varno += 1; v = "%__v$(varno)"
            println(body, "    $(v)s = stablehlo.slice %__arena [$(off):$(off+len)] : ($flatty) -> tensor<$(len)x$(elt)>")
            println(body, "    $(v) = stablehlo.reshape $(v)s : (tensor<$(len)x$(elt)>) -> $(_tystr(eltype(t), size(t)))")
            push!(callparams, v)
        end
    end
    calltys = "(" * join(callptys, ", ") * ") -> $outsig"
    if nres == 1
        println(body, "    %__r = func.call @__jolt_graph($(join(callparams, ", "))) : $calltys")
        println(body, "    return %__r : $(res_tys[1])")
    else
        println(body, "    %__r:$(nres) = func.call @__jolt_graph($(join(callparams, ", "))) : $calltys")
        println(body, "    return $(join(("%__r#$(k-1)" for k in 1:nres), ", ")) : $(join(res_tys, ", "))")
    end

    return string("module {\n",
        "  ", graph_func, "\n",
        "  func.func @main($fullsig) -> $outsig {\n",
        String(take!(body)),
        "  }\n}\n")
end

function compile(rng, inputs::Vector{<:AbstractTensor}, outputs::AbstractVector; backend::IREEBackend = IREE_CPU)
    all(t isa Argument for t in inputs) ||
        error("compile: `inputs` must be Argument tensors")
    s       = session()
    vars    = _materialize_vars(s, _seed_base(rng))
    argplan = _arg_plan(s, inputs)                        # before build_module finalizes the session
    layout  = _var_layout(s, vars)
    # A PackedGrad output contributes its flat tensor to the graph; `fn` rebuilds
    # it as a ComponentArray from the recorded axes. Plain tensors pass through.
    raw    = AbstractTensor[]
    repack = Any[]
    for o in outputs
        if o isa PackedGrad
            push!(raw, o.flat); push!(repack, o.axes)
        else
            push!(raw, o);      push!(repack, nothing)
        end
    end
    # `assign!`ed variables become EXTRA graph outputs (their new values), trailing
    # the user outputs; `fn` copies each back into its arena slot after the call.
    # Sorted by the target variable's registry path so the trailing order (and the
    # `state_paths` it must match) is deterministic.
    path_of     = IdDict(t => p for (p, t) in s.names)
    updates     = sort(collect(s.assigns); by = kv -> path_of[first(kv)])
    targets     = AbstractTensor[v  for (v, _)  in updates]  # variable written back
    newvals     = AbstractTensor[nv for (_, nv) in updates]  # its new value (graph output)
    state_paths = Any[path_of[v][2:end] for v in targets]    # CA path (drop the :variables head)
    has_vars    = !isempty(layout)
    text = if has_vars
        _arena_text(s, build_module(s, vcat(raw, newvals); entry = "__jolt_graph"),
                    raw, targets, vars, layout)
    else
        stablehlo_text(build_module(s, raw))              # no Variables → no arena arg
    end
    exe = iree_build(backend, text)
    exe.has_arena = has_vars
    d = getdata(vars)
    return CompiledFn(exe, argplan, length(outputs), state_paths, repack,
                      length(d), eltype(d), getaxes(vars)), vars
end

# `vars` may be any ComponentArray congruent with the compiled one (the same
# arena, a `snapshot`, or a fresh congruent CA) — its layout must match what the
# graph baked in. We check length + eltype + axes rather than object identity, so
# snapshots and populations work.
function _check_arena(f::CompiledFn, vars)
    f.exe.has_arena || return
    d = getdata(vars)
    (length(d) == f.arena_len && eltype(d) == f.arena_elt && getaxes(vars) == f.arena_axes) ||
        error("compile: `fn` needs a `vars` matching the compiled layout (length $(f.arena_len), " *
              "eltype $(f.arena_elt)). Pass the `vars` from `compile`, a `snapshot`, or a congruent " *
              "ComponentArray — not a reshaped, retyped, or differently-scoped one.")
end

function (f::CompiledFn)(vars, args...)
    _check_arena(f, vars)
    arena = f.exe.has_arena ? getdata(vars) : nothing
    outs = iree_run(f.exe, arena, Any[args[i] for i in f.argplan])
    # `assign!` updates trail the user outputs; copy each into its `vars` slot in
    # place (same arena ⇒ the next call reads the new value). Reads inside the graph
    # used the OLD value (the arena is read-only there), so this is read-old/write-new.
    for (k, capath) in enumerate(f.state_paths)
        sub = foldl(getproperty, capath; init = vars)
        sub .= outs[f.nout + k]
    end
    # rebuild any PackedGrad output as a ComponentArray matching its vars subtree
    user = Any[f.repack[i] === nothing ? outs[i] : ComponentArray(outs[i], f.repack[i]...) for i in 1:f.nout]
    return f.nout == 1 ? user[1] : Tuple(user)
end

# ---- vars lifecycle: snapshot / pure ---------------------------------------

"""
    snapshot(vars) -> ComponentArray

A deep copy of `vars` backed by a FRESH page-aligned arena (like the one
`compile` returns), so it stays eligible for zero-copy IREE import. Use this —
not `deepcopy` — to keep a checkpoint you can pass back to `fn`, or to seed a
population: `deepcopy` loses the aligned arena backing and would fall back to a
copy. `fn` advances any `assign!`ed variables IN the arena of whatever `vars` it
is handed, so `snapshot` first to preserve the current values.
"""
function snapshot(vars::ComponentArray)
    d = getdata(vars)
    isempty(d) && return ComponentArray(copy(d), getaxes(vars)...)
    arena = Mmap.mmap(Vector{eltype(d)}, length(d))       # anonymous ⇒ page-aligned
    copyto!(arena, d)
    return ComponentArray(arena, getaxes(vars)...)
end

"""
    y, new_vars = pure(fn, vars, args...)

Run `fn` functionally: snapshot `vars`, run `fn` on the snapshot, and return
`(outputs, snapshot)`. The `vars` you pass in is left UNCHANGED; `new_vars` holds
any advanced (`assign!`ed) values. This trades a whole-arena copy for
immutability — in a hot loop call `fn(vars, args...)` directly (zero-copy,
mutates in place).
"""
function pure(fn, vars::ComponentArray, args...)
    snap = snapshot(vars)
    y = fn(snap, args...)
    return y, snap
end

"The graph as StableHLO text — `compile` without a backend."
function export_stablehlo(inputs::Vector{<:AbstractTensor}, outputs::AbstractVector)
    all(t isa Argument for t in inputs) || error("export: `inputs` must be Argument tensors")
    raw = AbstractTensor[o isa PackedGrad ? o.flat : o for o in outputs]
    return stablehlo_text(build_module(session(), raw))
end
