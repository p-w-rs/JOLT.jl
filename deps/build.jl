# Runs at `Pkg.build`. Fetches iree-compile and builds the IREE runtime shim, caching both
# in JOLT's Scratch spaces. Non-fatal: if it fails (no network / no cmake+ninja), `using JOLT`
# still works — the backend sets up lazily on first compile and surfaces the same error there.
include(joinpath(@__DIR__, "iree_build.jl"))

# JOLT's package UUID (keep in sync with Project.toml) — the Scratch owner, so build-time and
# runtime resolve the SAME cache.
const JOLT_UUID = Base.UUID("987ac434-48f5-405e-b42f-6adc1ee8369d")

try
    IREEBuild.setup!(JOLT_UUID)
    @info "JOLT: iree-compile fetched and runtime shim built."
catch e
    @warn """JOLT: IREE setup during build failed. The package still loads; the backend retries \
             on first use. To use it, ensure network access plus `cmake`, `ninja`, and a C/C++ \
             compiler on PATH, then rerun `] build JOLT`.""" exception = (e, catch_backtrace())
end
