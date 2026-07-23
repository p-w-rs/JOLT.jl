# =====================================================================
# Graph → StableHLO module.
#
# A finished Session is a block of ops plus `argvars` — the Argument/Variable
# block arguments in call-signature order. To hand the graph to a backend (or
# hand it back to the user via `export`), we wrap that block in
#
#     func.func @main(argvars...) -> (outputs...)
#
# put the func in a module, and print it as StableHLO text. This is the single
# piece `compile` and `export` share, so it lives on its own.
# =====================================================================

# Wrap the session's graph in `func.func @main`. This FINALIZES the session:
# its block is moved into the returned module (and a `func.return` is appended),
# so the session should not be built into afterwards.
function build_module(s::Session, outputs::Vector{<:Tensor})
    in_tys  = IR.Type[IR.type(t.value) for t in s.argvars]        # inputs = args + vars, in order
    out_tys = IR.Type[IR.type(o.value) for o in outputs]
    push!(s.block, funcd.return_(IR.Value[o.value for o in outputs]))
    body = IR.Region(); push!(body, s.block)
    fn = funcd.func_(; sym_name = "main",
                       function_type = IR.FunctionType(in_tys, out_tys),
                       body = body)
    mod = IR.Module()
    push!(IR.body(mod), fn)
    return mod
end

# The module as StableHLO text — what `iree-compile` ingests and what `export`
# hands back.
stablehlo_text(mod::IR.Module) = string(IR.Operation(mod))
