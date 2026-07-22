# Shared IREE acquisition logic, used by BOTH `deps/build.jl` (at `Pkg.add`/`Pkg.build`
# time) and the runtime backend (as a lazy fallback). Products are cached in Scratch
# scratchspaces owned by the Myelin package — the installed package dir is read-only, so we
# never write there. Two halves:
#   • compiler: fetch the native `iree-compile` from IREE's GitHub release (a zip, no pip)
#   • runtime:  git-clone IREE and build our shim from source with cmake/ninja (no LLVM)
module IREEBuild

import Scratch, Libdl, Downloads, SHA, p7zip_jll

const VERSION = "3.12.0rc20260712"
const RELEASE = "https://github.com/iree-org/iree/releases/download/iree-$(VERSION)"

# ---- compiler (fetched native binary; run as a subprocess) ------------------------
function _compiler_wheel()
    if Sys.isapple()                      # universal2 wheel: arm64 + x86_64
        ("iree_base_compiler-$(VERSION)-cp312-abi3-macosx_13_0_universal2.whl",
         "63fea0f16f9317ecd0c64956f1aef599698b50231cc475b243bc22f2487902ca")
    else
        error("Myelin's bundled iree-compile is wired for macOS; on $(Sys.KERNEL)/$(Sys.ARCH) " *
              "set MYELIN_IREE_COMPILE or add the wheel to `_compiler_wheel()`.")
    end
end

compiler_dir(uuid) = Scratch.get_scratch!(uuid, "iree-compiler-$(VERSION)")
compiler_bin(uuid) = joinpath(compiler_dir(uuid), "iree", "compiler", "_mlir_libs", "iree-compile")

function fetch_compiler!(uuid)
    ic = compiler_bin(uuid)
    isfile(ic) && return ic
    whl, sha = _compiler_wheel()
    dir = compiler_dir(uuid)
    @info "Myelin: fetching iree-compile $(VERSION) from GitHub releases (one-time, ~72 MB)…"
    tmp = tempname() * ".zip"
    Downloads.download(joinpath(RELEASE, whl), tmp)
    bytes2hex(SHA.sha256(read(tmp))) == sha || error("iree-compile wheel sha256 mismatch")
    run(`$(p7zip_jll.p7zip()) x -tzip -y -o$dir $tmp`)
    rm(tmp; force=true)
    chmod(ic, 0o755)
    return ic
end

# ---- runtime (built from source into a clean shim; ccall'd in-process) ------------
runtime_dir(uuid) = Scratch.get_scratch!(uuid, "iree-runtime-$(VERSION)")
runtime_lib(uuid) = joinpath(runtime_dir(uuid), "libmyelin_iree.$(Libdl.dlext)")
shim_source()     = joinpath(@__DIR__, "iree_shim")     # committed shim.c + CMakeLists

function build_runtime!(uuid; shimdir=shim_source())
    lib = runtime_lib(uuid)
    isfile(lib) && return lib
    cmake = Sys.which("cmake"); ninja = Sys.which("ninja")
    (cmake !== nothing && ninja !== nothing) ||
        error("Building the IREE runtime shim needs `cmake` and `ninja` on PATH (plus a C/C++ compiler).")
    cache = runtime_dir(uuid); src = joinpath(cache, "iree_src")
    @info "Myelin: building the IREE runtime shim from source (one-time: git clone IREE + cmake/ninja)…"
    if !isdir(joinpath(src, "runtime"))
        run(`git clone --depth 1 -b iree-$(VERSION) https://github.com/iree-org/iree.git $src`)
        gm = joinpath(src, ".gitmodules")
        for ln in eachline(`git -C $src config -f $gm --get-regexp path`)
            path = split(ln)[2]
            path in ("third_party/llvm-project", "third_party/torch-mlir", "third_party/stablehlo") && continue
            run(`git -C $src submodule update --init --depth 1 $path`)
        end
    end
    build = joinpath(cache, "build")
    run(`$cmake -G Ninja -B $build -S $shimdir -DIREE_SRC=$src -DCMAKE_BUILD_TYPE=Release`)
    run(`$cmake --build $build --target myelin_iree -j $(Sys.CPU_THREADS)`)
    cp(joinpath(build, "libmyelin_iree.$(Libdl.dlext)"), lib; force=true)
    return lib
end

"Fetch the compiler and build the runtime shim (idempotent; used by deps/build.jl)."
function setup!(uuid)
    fetch_compiler!(uuid)
    build_runtime!(uuid)
    return nothing
end

end # module IREEBuild
