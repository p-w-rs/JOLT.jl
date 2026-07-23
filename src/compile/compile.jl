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
# D-flat weights (io_parameters): Variables are NOT compiled-in as func inputs.
# The whole `vars` arena is served as ONE runtime parameter (scope::key), and the
# graph slices each Variable out of it — so `vars` stays a dense ComponentArray
# and IREE reads it zero-copy (see the shim + [[jolt-verified-constraints]]).
# ====================================================================
const _PARAM_SCOPE = "jolt"
const _PARAM_KEY   = "params"

# MLIR spellings, matching module.jl's mlir_type (REVERSE-DIMS: Julia (m,n) → nxm).
_eltstr(::Type{Float32}) = "f32"; _eltstr(::Type{Float64}) = "f64"; _eltstr(::Type{Float16}) = "f16"
_eltstr(::Type{Int32}) = "i32"; _eltstr(::Type{Int64}) = "i64"; _eltstr(::Type{Int8}) = "i8"; _eltstr(::Type{Bool}) = "i1"
_dimstr(d) = d isa Integer ? string(d) : "?"        # symbolic/dynamic dim → MLIR `?` (cf. dim_to_mlir)
_tystr(::Type{T}, shape) where {T} =
    isempty(shape) ? "tensor<$(_eltstr(T))>" :
    "tensor<$(join((_dimstr(d) for d in reverse(shape)), 'x'))x$(_eltstr(T))>"

struct CompiledFn
    exe::IREEExecutable
    argplan::Vector{Int}    # compiled @main input i ← value of call arg #argplan[i]
    nout::Int
end

# The compiled @main takes ONLY the Arguments (Variables come via the provider).
# For each Argument in graph/block order, the index into the user's `inputs`.
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

# Assemble the D-flat module text: the graph (as private @__jolt_graph) plus a
# util.global weights parameter and a @main wrapper that loads it, inserts an
# optimization_barrier (WORKAROUND: iree-compile 3.12 segfaults folding a
# flow.tensor.slice whose source is a #flow.parameter — see test/compile.jl and
# the filed IREE issue), slices each Variable out of the flat arena, and calls the
# graph. Variables reach the graph via the provider, so @main's inputs are only
# the Arguments.
function _dflat_text(s::Session, graph_mod::IR.Module, outputs::Vector{<:AbstractTensor},
                     vars::ComponentArray, layout::Dict{Tuple,Tuple{Int,Int}})
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
    outtys = [_tystr(eltype(o), size(o)) for o in outputs]
    outsig = length(outtys) == 1 ? outtys[1] : "(" * join(outtys, ", ") * ")"

    body = IOBuffer()
    println(body, "    %__flat0 = util.global.load @jolt_params : $flatty")
    println(body, "    %__flat = stablehlo.optimization_barrier %__flat0 : $flatty")
    callparams = String[]; callptys = String[]; varno = 0
    for t in s.argvars
        push!(callptys, _tystr(eltype(t), size(t)))
        if t isa Argument
            push!(callparams, "%a$(argno[t])")
        else
            off, len = layout[path_of[t][2:end]]
            v = "%__v$(varno)"; varno += 1
            println(body, "    $(v)s = stablehlo.slice %__flat [$(off):$(off+len)] : ($flatty) -> tensor<$(len)x$(elt)>")
            println(body, "    $(v) = stablehlo.reshape $(v)s : (tensor<$(len)x$(elt)>) -> $(_tystr(eltype(t), size(t)))")
            push!(callparams, v)
        end
    end
    calltys = "(" * join(callptys, ", ") * ") -> $outsig"
    if length(outtys) == 1
        println(body, "    %__r = func.call @__jolt_graph($(join(callparams, ", "))) : $calltys")
        println(body, "    return %__r : $(outtys[1])")
    else
        println(body, "    %__r:$(length(outtys)) = func.call @__jolt_graph($(join(callparams, ", "))) : $calltys")
        println(body, "    return $(join(("%__r#$(k-1)" for k in 1:length(outtys)), ", ")) : $(join(outtys, ", "))")
    end

    return string("module {\n",
        "  util.global private @jolt_params = #flow.parameter.named<\"$(_PARAM_SCOPE)\"::\"$(_PARAM_KEY)\"> : $flatty\n",
        "  ", graph_func, "\n",
        "  func.func @main($argsig) -> $outsig {\n",
        String(take!(body)),
        "  }\n}\n")
end

function compile(rng, inputs::Vector{<:AbstractTensor}, outputs::Vector{<:AbstractTensor}; backend::IREEBackend = IREE_CPU)
    all(t isa Argument for t in inputs) ||
        error("compile: `inputs` must be Argument tensors")
    s       = session()
    vars    = _materialize_vars(s, _seed_base(rng))
    argplan = _arg_plan(s, inputs)                        # before build_module finalizes the session
    layout  = _var_layout(s, vars)
    text = if isempty(layout)
        stablehlo_text(build_module(s, outputs))          # no Variables → plain graph, no provider
    else
        _dflat_text(s, build_module(s, outputs; entry = "__jolt_graph"), outputs, vars, layout)
    end
    exe = iree_build(backend, text)
    isempty(layout) || bind_params!(exe, getdata(vars), _PARAM_SCOPE, _PARAM_KEY)
    return CompiledFn(exe, argplan, length(outputs)), vars
end

function (f::CompiledFn)(vars, args...)
    e = f.exe
    if e.param_nbytes > 0 && pointer(getdata(vars)) != e.param_ptr
        error("compile: `fn` must be called with the SAME `vars` ComponentArray it was compiled with " *
              "(its arena is shared with IREE zero-copy). Mutate `vars` in place (e.g. Optimisers.update!); " *
              "do not reallocate it or pass a copy.")
    end
    outs = iree_run(e, Any[args[i] for i in f.argplan])
    return f.nout == 1 ? outs[1] : Tuple(outs)
end

"The graph as StableHLO text — `compile` without a backend."
export_stablehlo(inputs::Vector{<:AbstractTensor}, outputs::Vector{<:AbstractTensor}) =
    (all(t isa Argument for t in inputs) || error("export: `inputs` must be Argument tensors");
     stablehlo_text(build_module(session(), outputs)))
