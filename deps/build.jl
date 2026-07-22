# Runs at `Pkg.add` / `Pkg.build`. Fetches iree-compile and builds the IREE runtime shim
# from source, caching both in Myelin's Scratch scratchspaces. Non-fatal: if this fails
# (no network / no cmake+ninja), `using Myelin` still works — the backend sets up lazily on
# first compile/run and surfaces the same error there.
include(joinpath(@__DIR__, "iree_build.jl"))

# Myelin's package UUID (keep in sync with Project.toml) — identifies the Scratch owner so
# build-time and runtime resolve the SAME cache.
const MYELIN_UUID = Base.UUID("987ac434-48f5-405e-b42f-6adc1ee8369d")

try
    IREEBuild.setup!(MYELIN_UUID)
    @info "Myelin: IREE compiler fetched and runtime shim built."
catch e
    @warn """Myelin: IREE setup during build failed. The package will still load; the IREE \
             backend will retry on first use. To use it, ensure network access plus `cmake`, \
             `ninja`, and a C/C++ compiler on PATH, then rerun `] build Myelin`.""" exception = (e, catch_backtrace())
end
