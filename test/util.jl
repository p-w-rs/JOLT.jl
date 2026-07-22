# =====================================================================
# Shared test helpers.
# =====================================================================

# Read the MLIR ranked-tensor type behind a Tensor's SSA value back out as
# (julia eltype, dims). Symbolic dims come back as the marker `:dyn` — the IR
# only knows them as anonymous `?`, so we can't recover the DimVar name here.
# Lets a test assert the emitted IR matches the Julia-side shape/dtype, not just
# that the Tensor handle claims to.
function mlir_ranked(t)
    ty = JOLT.IR.type(t.value)
    @assert JOLT.IR.hasrank(ty) "expected a ranked tensor type"
    T = JOLT.IR.julia_type(JOLT.IR.eltype(ty))
    dims = ntuple(JOLT.IR.ndims(ty)) do i
        sz = JOLT.IR.size(ty, i)
        JOLT.IR.isdynsize(sz) ? :dyn : Int(sz)
    end
    return T, dims
end
