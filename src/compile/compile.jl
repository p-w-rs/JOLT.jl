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
        roleof(t) === Variable || continue
        arr = initializer(t)(variable_rng(base, path), eltype(t), size(t)...)
        push!(pairs, path[2:end] => arr)              # drop the :variables head
    end
    return ComponentArray(_nest(pairs))
end

# ---- binding plan: for each func input, where its value comes from at call ----
struct CompiledFn
    exe::IREEExecutable
    plan::Vector{Tuple{Symbol,Any}}    # (:var, nested-path) or (:arg, call-index)
    nout::Int
end

function _io_plan(s::Session, inputs::Vector{<:Tensor})
    path_of = Dict(t => p for (p, t) in s.names)
    argidx  = IdDict(t => i for (i, t) in enumerate(inputs))
    return Tuple{Symbol,Any}[
        if roleof(t) === Variable
            (:var, path_of[t][2:end])
        else
            haskey(argidx, t) || error("compile: an Argument feeds the graph but isn't in `inputs`")
            (:arg, argidx[t])
        end
        for t in s.argvars]
end

function compile(rng, inputs::Vector{<:Tensor}, outputs::Vector{<:Tensor}; backend::IREEBackend = IREE_CPU)
    all(roleof(t) === Argument for t in inputs) ||
        error("compile: `inputs` must be Argument tensors")
    s    = session()
    vars = _materialize_vars(s, _seed_base(rng))
    plan = _io_plan(s, inputs)                        # before build_module finalizes the session
    exe  = iree_build(backend, stablehlo_text(build_module(s, outputs)))
    return CompiledFn(exe, plan, length(outputs)), vars
end

_getpath(vars, path::Tuple) = foldl(getproperty, path; init = vars)   # vars.params.W
function (f::CompiledFn)(vars, args...)
    ins  = Any[kind === :var ? _getpath(vars, ref) : args[ref] for (kind, ref) in f.plan]
    outs = iree_run(f.exe, ins)
    return f.nout == 1 ? outs[1] : Tuple(outs)
end

"The graph as StableHLO text — `compile` without a backend."
export_stablehlo(inputs::Vector{<:Tensor}, outputs::Vector{<:Tensor}) =
    (all(roleof(t) === Argument for t in inputs) || error("export: `inputs` must be Argument tensors");
     stablehlo_text(build_module(session(), outputs)))
