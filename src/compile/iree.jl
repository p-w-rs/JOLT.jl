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
    for f in (:jolt_session_create, :jolt_session_release, :jolt_import_probe,
              :jolt_call_begin, :jolt_push_arena, :jolt_push_input, :jolt_invoke,
              :jolt_output_next, :jolt_output_read, :jolt_call_end)
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
    # The `vars` arena rides as READ-ONLY call input 0 (see compile.jl / shim.c):
    # the graph slices each Variable out of it, and `assign!` updates come back as
    # separate outputs that `fn` copies into the arena. `has_arena` ⇒ the graph
    # takes it. `zerocopy` is the per-session import probe: -1 not yet probed, 1 the
    # device reads the arena zero-copy (CPU heap / Metal shared) — a Julia-side
    # mutation between calls is then seen live; 0 the arena is copied in per call
    # (e.g. a discrete GPU that can't alias host memory).
    has_arena::Bool
    zerocopy::Int
end

"Compile StableHLO `text` to a .vmfb for `backend`."
function iree_build(backend::IREEBackend, text::AbstractString)
    _ensure_runtime()
    dir = mktempdir(); inp = joinpath(dir, "in.mlir"); vmfb = joinpath(dir, "out.vmfb")
    write(inp, text)
    # Respect the user's declared dtype: iree-compile demotes f64→f32 and i64→i32
    # by DEFAULT, which would silently lose precision. Keep the declared type; a
    # backend that can't support it (e.g. f64 on Metal) then errors honestly.
    run(`$(_iree_compile()) --iree-input-type=stablehlo
         --iree-input-demote-f64-to-f32=false --iree-input-demote-i64-to-i32=false
         --iree-hal-target-backends=$(backend.target) $(backend.cflags) $inp -o $vmfb`)
    return IREEExecutable(vmfb, backend.driver, C_NULL, false, -1)
end

# Contiguous column-major bytes of `a`. With the reverse-dims graph these bytes
# ARE the row-major tensor IREE wants — no transpose. (Dense arrays share memory;
# views are materialized. The zero-copy shim will import a dense array's pointer.)
_flat(a::Array)         = vec(a)
_flat(a::AbstractArray) = vec(Array(a))

"""
Run the program → output host arrays (user outputs, then each `assign!`ed
variable's new value). `arena` is the `vars` ComponentArray's contiguous backing
(or `nothing` for a graph with no Variables): it is pushed as READ-ONLY call
input 0 — imported zero-copy where the device allows (so a Julia-side mutation
between calls is seen live), else copied in. `inputs` are the activation values
in @main Argument order (copied in). `fn` copies the trailing state outputs back
into `arena`'s slots.
"""
function iree_run(e::IREEExecutable, arena, inputs::Vector)
    _ensure_runtime()
    has_arena = e.has_arena && arena !== nothing && !isempty(arena)
    if e.session == C_NULL
        e.session = ccall(_sym[:jolt_session_create], Ptr{Cvoid},
            (Cstring, Cstring), e.vmfb, e.driver)
        e.session == C_NULL && error("IREE: session create failed for $(e.vmfb) on driver `$(e.driver)`")
        # Probe once: can this device import the arena zero-copy (read live)?
        if has_arena
            e.zerocopy = GC.@preserve arena Int(ccall(_sym[:jolt_import_probe], Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Int64), e.session, pointer(arena), Int64(sizeof(arena))))
        else
            e.zerocopy = 0
        end
    end
    GC.@preserve arena begin
        call = ccall(_sym[:jolt_call_begin], Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), e.session, "module.main")
        call == C_NULL && error("IREE: call_begin failed for entry `module.main`")
        try
            # arena as read-only input 0: zero-copy import where supported, else copy-in.
            if has_arena
                adims = Int64[length(arena)]          # a flat 1-D tensor (reverse of [n] is [n])
                m = GC.@preserve adims Int(ccall(_sym[:jolt_push_arena], Cint,
                    (Ptr{Cvoid}, Cint, Cint, Ptr{Int64}, Ptr{Cvoid}, Int64, Cint),
                    call, Cint(_ecode(eltype(arena))), Cint(1), pointer(adims),
                    pointer(arena), Int64(sizeof(arena)), Cint(e.zerocopy == 1 ? 1 : 0)))
                m < 0 && error("IREE: push_arena failed")
            end
            # activation inputs (copied in), in @main Argument order.
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
            ec = Ref{Cint}(0); rk = Ref{Cint}(0); dims = zeros(Int64, 16)
            outs = Any[]
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
end
