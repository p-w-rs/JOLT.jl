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
    target::String     # iree-compile  --iree-hal-target-backends=…
    driver::String     # runtime HAL driver
end
const IREE_CPU   = IREEBackend("llvm-cpu",    "local-task")
const IREE_METAL = IREEBackend("metal-spirv", "metal")

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
end

"Compile StableHLO `text` to a .vmfb for `backend`."
function iree_build(backend::IREEBackend, text::AbstractString)
    _ensure_runtime()
    dir = mktempdir(); inp = joinpath(dir, "in.mlir"); vmfb = joinpath(dir, "out.vmfb")
    write(inp, text)
    run(`$(_iree_compile()) --iree-input-type=stablehlo --iree-hal-target-backends=$(backend.target) $inp -o $vmfb`)
    return IREEExecutable(vmfb, backend.driver, C_NULL)
end

# Column-major Julia array → row-major flat bytes (and back). This transpose is
# the boundary copy the zero-copy phase eliminates.
_rowmajor(a::AbstractArray)          = vec(permutedims(a, reverse(1:ndims(a))))
_rowmajor(a::AbstractArray{<:Any,0}) = collect(vec(a))

"Run the program on `inputs` (host arrays, in func-signature order) → output host arrays."
function iree_run(e::IREEExecutable, inputs::Vector)
    _ensure_runtime()
    if e.session == C_NULL
        e.session = ccall(_sym[:jolt_session_create], Ptr{Cvoid}, (Cstring, Cstring), e.vmfb, e.driver)
        e.session == C_NULL && error("IREE: session create failed for $(e.vmfb) on driver `$(e.driver)`")
    end
    call = ccall(_sym[:jolt_call_begin], Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), e.session, "module.main")
    call == C_NULL && error("IREE: call_begin failed for entry `module.main`")
    try
        for x in inputs
            arr = x isa AbstractArray ? x : fill(x)
            data = _rowmajor(arr); dims = Int64[size(arr)...]
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
            push!(outs, r == 0 ? buf[1] : permutedims(reshape(buf, reverse(shp)...), reverse(1:r)))
        end
        return outs
    finally
        ccall(_sym[:jolt_call_end], Cvoid, (Ptr{Cvoid},), call)
    end
end
