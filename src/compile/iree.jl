# =====================================================================
# IREE backend — everything that talks to IREE lives here.
#
# Two halves, mirroring how IREE ships (see deps/iree_build.jl):
#   • iree_build : StableHLO text  --(iree-compile subprocess)-->  a .vmfb
#   • iree_run   : run a .vmfb in-process via the ccall shim (libjolt_iree)
#
# A `IREEBackend` just names the compile target and the runtime HAL driver, so
# a new device is one line (see the CPU/Metal presets). The layout reconciliation
# (Julia column-major ⇄ IREE row-major) is done here at the boundary; the
# zero-copy phase moves it into the graph (reverse-dims) + a host-buffer import.
# =====================================================================

include(joinpath(@__DIR__, "..", "..", "deps", "iree_build.jl"))   # module IREEBuild
const _UUID = Base.UUID("987ac434-48f5-405e-b42f-6adc1ee8369d")

struct IREEBackend
    target::String          # iree-compile  --iree-hal-target-backends=…
    driver::String          # runtime HAL driver
    cflags::Vector{String}  # extra iree-compile flags for this target
end
IREEBackend(target, driver) = IREEBackend(target, driver, String[])
const IREE_CPU   = IREEBackend("llvm-cpu",    "local-task")
# Metal: embed MSL source and let the runtime compile it via Metal.framework at
# load — so we DON'T need Xcode's `metal`/`metallib` CLI (Command Line Tools lack
# them). `newLibraryWithSource` runs on the device at session-create instead.
const IREE_METAL = IREEBackend("metal-spirv", "metal", ["--iree-metal-compile-to-metallib=false"])

# ---- element-type codes, shared with shim.c --------------------------------
_ecode(::Type{Float32}) = 0; _ecode(::Type{Float64}) = 1; _ecode(::Type{Float16}) = 2
_ecode(::Type{Int32})   = 3; _ecode(::Type{Int64})   = 4; _ecode(::Type{Int8})    = 5
_ecode(::Type{Bool})    = 6
const _CODE2T = (Float32, Float64, Float16, Int32, Int64, Int8, Bool)

# ---- the runtime shim: dlopen once, resolve the jolt_* symbols -------------
const _lib = Ref{Ptr{Cvoid}}(C_NULL)
const _sym = Dict{Symbol,Ptr{Cvoid}}()
function _ensure_runtime()
    _lib[] == C_NULL || return
    lib = IREEBuild.runtime_lib(_UUID)
    isfile(lib) || IREEBuild.build_runtime!(_UUID)
    _lib[] = Libdl.dlopen(lib)
    for f in (:jolt_session_create, :jolt_session_release, :jolt_call_begin, :jolt_push_input,
              :jolt_invoke, :jolt_output_next, :jolt_output_read, :jolt_call_end)
        _sym[f] = Libdl.dlsym(_lib[], f)
    end
    return
end
_iree_compile() = (b = IREEBuild.compiler_bin(_UUID); isfile(b) || IREEBuild.fetch_compiler!(_UUID); b)

# ---- a compiled program: a .vmfb + a lazily-opened runtime session ---------
mutable struct IREEExecutable
    vmfb::String
    driver::String
    session::Ptr{Cvoid}
    # Zero-copy weights (io_parameters): the session's provider ALIASES this arena
    # (the `vars` ComponentArray's backing). `param_hold` keeps it GC-alive for the
    # session's lifetime — the import is non-owning. `param_nbytes == 0` ⇒ no
    # provider (a graph with no variables, or a non-IREE path).
    param_ptr::Ptr{Cvoid}
    param_nbytes::Int
    param_hold::Any
    scope::String
    key::String
end

"Compile StableHLO `text` to a .vmfb for `backend`."
function iree_build(backend::IREEBackend, text::AbstractString)
    _ensure_runtime()
    dir = mktempdir(); inp = joinpath(dir, "in.mlir"); vmfb = joinpath(dir, "out.vmfb")
    write(inp, text)
    # Respect the user's declared dtype: iree-compile demotes f64→f32 and i64→i32
    # by DEFAULT, which would silently lose precision (and breaks our parameter
    # global, whose value attr keeps the original type). Keep the declared type;
    # a backend that can't support it (e.g. f64 on Metal) then errors honestly.
    run(`$(_iree_compile()) --iree-input-type=stablehlo
         --iree-input-demote-f64-to-f32=false --iree-input-demote-i64-to-i32=false
         --iree-hal-target-backends=$(backend.target) $(backend.cflags) $inp -o $vmfb`)
    return IREEExecutable(vmfb, backend.driver, C_NULL, C_NULL, 0, nothing, "", "")
end

# Bind `vars`' contiguous arena as the session's zero-copy weights parameter.
# Called by `compile` once the ComponentArray exists; the executable then keeps
# the arena alive and hands its pointer to the provider at session-create.
function bind_params!(e::IREEExecutable, data::AbstractVector, scope::AbstractString, key::AbstractString)
    e.param_hold   = data
    e.param_ptr    = pointer(data)
    e.param_nbytes = sizeof(data)
    e.scope        = String(scope)
    e.key          = String(key)
    return e
end

# Contiguous column-major bytes of `a`. With the reverse-dims graph these bytes
# ARE the row-major tensor IREE wants — no transpose. (Dense arrays share memory;
# views are materialized. The zero-copy shim will import a dense array's pointer.)
_flat(a::Array)         = vec(a)
_flat(a::AbstractArray) = vec(Array(a))

"Run the program on `inputs` (host arrays, in func-signature order) → output host arrays."
function iree_run(e::IREEExecutable, inputs::Vector)
    _ensure_runtime()
    if e.session == C_NULL
        e.session = GC.@preserve e ccall(_sym[:jolt_session_create], Ptr{Cvoid},
            (Cstring, Cstring, Cstring, Cstring, Ptr{Cvoid}, Int64),
            e.vmfb, e.driver, e.scope, e.key, e.param_ptr, Int64(e.param_nbytes))
        e.session == C_NULL && error("IREE: session create failed for $(e.vmfb) on driver `$(e.driver)`")
    end
    call = ccall(_sym[:jolt_call_begin], Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), e.session, "module.main")
    call == C_NULL && error("IREE: call_begin failed for entry `module.main`")
    try
        for x in inputs
            arr = x isa AbstractArray ? x : fill(x)
            data = _flat(arr); dims = Int64[reverse(size(arr))...]   # reversed dims (see mlir_type)
            rc = GC.@preserve data dims ccall(_sym[:jolt_push_input], Cint,
                (Ptr{Cvoid}, Cint, Cint, Ptr{Int64}, Ptr{Cvoid}, Int64),
                call, Cint(_ecode(eltype(arr))), Cint(ndims(arr)),
                pointer(dims), pointer(data), Int64(sizeof(data)))
            rc == 0 || error("IREE: push_input failed (rc=$rc)")
        end
        ccall(_sym[:jolt_invoke], Cint, (Ptr{Cvoid},), call) == 0 || error("IREE: invoke failed")
        outs = Any[]
        ec = Ref{Cint}(0); rk = Ref{Cint}(0); dims = zeros(Int64, 16)
        while (GC.@preserve dims ccall(_sym[:jolt_output_next], Cint,
                (Ptr{Cvoid}, Ref{Cint}, Ref{Cint}, Ptr{Int64}), call, ec, rk, pointer(dims))) == 0
            T = _CODE2T[Int(ec[]) + 1]; r = Int(rk[]); shp = ntuple(i -> Int(dims[i]), r)
            buf = Vector{T}(undef, prod(shp; init = 1))
            GC.@preserve buf ccall(_sym[:jolt_output_read], Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Int64), call, pointer(buf), Int64(sizeof(buf))) == 0 ||
                error("IREE: output_read failed")
            # IREE reports the reversed shape; reshape the row-major bytes to the Julia
            # shape (its reverse) — a column-major reshape reinterprets them, no transpose.
            push!(outs, r == 0 ? buf[1] : reshape(buf, reverse(shp)...))
        end
        return outs
    finally
        ccall(_sym[:jolt_call_end], Cvoid, (Ptr{Cvoid},), call)
    end
end
