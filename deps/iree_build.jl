# =====================================================================
# Acquiring IREE (no Python).
#
# JOLT talks to IREE two ways, and this module produces both, caching them in
# per-package Scratch spaces (the installed package dir is read-only):
#
#   • compiler — the native `iree-compile`, run as a subprocess to turn our
#     StableHLO into a `.vmfb`. We don't build it (that needs LLVM); we just
#     download IREE's official release archive (a `.whl`, which is a plain zip)
#     and extract the binary with p7zip. No pip, no Python is ever executed.
#
#   • runtime  — a tiny C shim (deps/iree_shim/shim.c) linked against
#     `iree::runtime` into one shared library that Julia `ccall`s in-process.
#     The runtime needs no LLVM, so we build it from a shallow IREE clone with
#     cmake + ninja.
#
# Used by both deps/build.jl (at install time) and, as a lazy fallback, the
# backend on first use — so a missing build surfaces the same clear error.
# =====================================================================
module IREEBuild

import Scratch, Libdl, Downloads, SHA, p7zip_jll

const VERSION = "3.12.0rc20260712"
const RELEASE = "https://github.com/iree-org/iree/releases/download/iree-$(VERSION)"

# ---- compiler: fetch the native iree-compile from the release archive --------------
# (platform, archive-name, sha256). Add rows as new platforms are supported.
function _compiler_archive()
    if Sys.isapple()
        ("iree_base_compiler-$(VERSION)-cp312-abi3-macosx_13_0_universal2.whl",
         "63fea0f16f9317ecd0c64956f1aef599698b50231cc475b243bc22f2487902ca")
    else
        error("JOLT's bundled iree-compile is wired for macOS; on $(Sys.KERNEL)/$(Sys.ARCH) " *
              "add the release archive to `_compiler_archive()`.")
    end
end

compiler_dir(uuid) = Scratch.get_scratch!(uuid, "iree-compiler-$(VERSION)")
compiler_bin(uuid) = joinpath(compiler_dir(uuid), "iree", "compiler", "_mlir_libs", "iree-compile")

function fetch_compiler!(uuid)
    bin = compiler_bin(uuid)
    isfile(bin) && return bin
    archive, sha = _compiler_archive()
    @info "JOLT: fetching iree-compile $(VERSION) from GitHub releases (one-time, ~72 MB)…"
    tmp = tempname() * ".zip"
    Downloads.download(joinpath(RELEASE, archive), tmp)
    bytes2hex(SHA.sha256(read(tmp))) == sha || error("iree-compile archive sha256 mismatch")
    run(`$(p7zip_jll.p7zip()) x -tzip -y -o$(compiler_dir(uuid)) $tmp`)
    rm(tmp; force = true)
    chmod(bin, 0o755)
    return bin
end

# ---- runtime: build the ccall shim from a shallow IREE clone -----------------------
runtime_dir(uuid) = Scratch.get_scratch!(uuid, "iree-runtime-$(VERSION)")
runtime_lib(uuid) = joinpath(runtime_dir(uuid), "libjolt_iree.$(Libdl.dlext)")
shim_dir()        = joinpath(@__DIR__, "iree_shim")     # committed shim.c + CMakeLists.txt

function build_runtime!(uuid; shimdir = shim_dir())
    lib = runtime_lib(uuid)
    isfile(lib) && return lib
    cmake = Sys.which("cmake"); ninja = Sys.which("ninja")
    (cmake !== nothing && ninja !== nothing) ||
        error("Building the IREE runtime shim needs `cmake` and `ninja` on PATH (plus a C/C++ compiler).")
    cache = runtime_dir(uuid); src = joinpath(cache, "iree_src")
    if !isdir(joinpath(src, "runtime"))
        @info "JOLT: cloning IREE $(VERSION) (shallow; skips the LLVM submodules the runtime doesn't need)…"
        run(`git clone --depth 1 -b iree-$(VERSION) https://github.com/iree-org/iree.git $src`)
        gm = joinpath(src, ".gitmodules")
        for ln in eachline(`git -C $src config -f $gm --get-regexp path`)
            path = split(ln)[2]
            path in ("third_party/llvm-project", "third_party/torch-mlir", "third_party/stablehlo") && continue
            run(`git -C $src submodule update --init --depth 1 $path`)
        end
    end
    @info "JOLT: building the IREE runtime shim (cmake + ninja, one-time)…"
    build = joinpath(cache, "build")
    run(`$cmake -G Ninja -B $build -S $shimdir -DIREE_SRC=$src -DCMAKE_BUILD_TYPE=Release`)
    run(`$cmake --build $build --target jolt_iree -j $(Sys.CPU_THREADS)`)
    cp(joinpath(build, "libjolt_iree.$(Libdl.dlext)"), lib; force = true)
    return lib
end

"Fetch the compiler and build the runtime shim (idempotent)."
setup!(uuid) = (fetch_compiler!(uuid); build_runtime!(uuid); nothing)

end # module IREEBuild
