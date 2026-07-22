# =====================================================================
# Session
#
# Holds the MLIR context and the block we accumulate ops into, plus the name
# registry. `mod` is filled at build time. `argvars` records Argument/Variable
# tensors in block-argument order (that ordering becomes the compiled graph's
# input signature). A Variable's initial/current value is read straight from
# its own DataBox, so there's no separate init map to keep in sync.
# =====================================================================
mutable struct Session
    context::IR.Context
    block::IR.Block
    argvars::Vector{Tensor}                     # Arg/Var tensors, in block-argument order
    scope::Vector{Symbol}                       # current namespace stack (mutable)
    names::Dict{Tuple{Vararg{Symbol}},Tensor}   # scoped path -> tensor
    anon::Dict{Symbol,Base.RefValue{Int}}             # monotonic counters for anonymous names
    outputs::Vector{Tensor}                     # set when the graph is finalized
    mod::Union{Nothing,IR.Module}
    facts::Facts                                # symbolic-dim constraints (dims.jl)
    tape::Vector{Tuple{OpNode,Tensor}}          # (node, output), in emission order — the AD tape
end

function Session()
    ctx   = Reactant.ReactantContext()
    block = IR.Block()
    return Session(
        ctx, block,
        Tensor[],
        Symbol[],
        Dict{Tuple{Vararg{Symbol}},Tensor}(),
        Dict{Symbol,Base.RefValue{Int}}(),
        Tensor[],
        nothing,
        Facts(),
        Tuple{OpNode,Tensor}[],
    )
end

function Base.show(io::IO, s::Session)
    print(io, "Session(", length(s.argvars), " inputs, ",
          length(s.names), " named, ", length(s.outputs), " outputs)")
end

# =====================================================================
# Default element type.
#
# The dtype a constructor falls back to when it isn't handed one (seeded
# from DEFAULT_ELTYPE in tensor.jl). Settable at runtime.
#
#   default_dtype()    -> the element type used when a constructor is given none
#   default_dtype!(T)  -> change it; persists until set again
# =====================================================================
const _DTYPE   = Base.RefValue{DataType}(DEFAULT_ELTYPE)
default_dtype() = _DTYPE[]
default_dtype!(::Type{T}) where {T<:Real} = (_DTYPE[] = T; T)

# =====================================================================
# Global session (transparent: you never have to create one).
#
#   session()          -> the active session (building + activating one on first use)
#   session!(s)        -> install `s` as the active session (activating its context)
#   new_session!()     -> fresh session, installed and returned
#   reset_session!()   -> drop the current session and pop its context
#   with_session(f, s) -> run `f` with `s` active, then restore the previous one
# =====================================================================
const _SESSION = Base.RefValue{Union{Nothing,Session}}(nothing)

# Make `s` current: pop whatever context is active, push s's onto the MLIR
# task-local stack, and record it. The _SESSION ref and the active MLIR context
# move together, so "current session" and "active context" can never disagree.
function session!(s::Session)
    old = _SESSION[]
    old === nothing || IR.deactivate(old.context)
    IR.activate(s.context)
    _SESSION[] = s
    return s
end

# The active session, building (and activating) a default on first use.
function session()
    _SESSION[] === nothing && session!(Session())
    return _SESSION[]::Session
end

# The active session's constraint store — the single point through which
# dims.jl's convenience queries (dims_equal, same_dim!, …) reach the session.
current_facts() = session().facts

new_session!() = session!(Session())

# Drop the current session and pop its context. Afterwards `session()` builds
# and activates a fresh default. Use between tests for a known-clean state.
function reset_session!()
    old = _SESSION[]
    old === nothing || IR.deactivate(old.context)
    _SESSION[] = nothing
    return nothing
end

# Run `f` with `s` active, then restore the previous session and its context.
# The finally block deactivates whatever session is CURRENT (not necessarily
# `s`): if `f` itself swapped sessions via session!/new_session!, the invariant
# "installed session's context is top of stack" still holds and must be used.
function with_session(f, s::Session)
    old = _SESSION[]
    old === nothing || IR.deactivate(old.context)
    IR.activate(s.context)
    _SESSION[] = s
    try
        return f()
    finally
        cur = _SESSION[]
        cur === nothing || IR.deactivate(cur.context)
        _SESSION[] = old
        old === nothing || IR.activate(old.context)
    end
end

# =====================================================================
# Naming, scopes & the name registry
#
# Every tensor is registered under a PATH of the form
#
#       (role, scope..., leaf)
#
#   role   — the tensor's kind, pluralized: :variables, :arguments,
#            :constants, :results. Role is the TOP level on purpose: a query
#            (e.g. a gradient) taken "wrt a top-level scope" then includes
#            everything registered beneath it, for free.
#   scope  — the active namespace stack, or (:default,) when the stack is
#            empty. Each nested namespace contributes one Symbol.
#   leaf   — the name you passed, or an anonymous `_N` when you passed none.
#            `N` is a per-role monotonic counter (s.anon[role]), so anonymous
#            leaves are unique across ALL scopes of that role and never clash.
#
# Examples:
#       Tensor(Var, 8)                 ->  (:variables, :default, :_1)
#       x + y   (a Result)             ->  (:results,   :default, :_1)
#       namespace("params") do
#           Tensor(Var, 8)             ->  (:variables, :params,  :_2)
#           Tensor(Var, 8, name="W")   ->  (:variables, :params,  :W)
#       end
#
# The scope is a plain MUTABLE stack (not a ScopedValue). Drive it directly
# with pushnamespace!/popnamespace!/clearnamespace!, or — preferred — with the
# `namespace(name) do … end` wrapper, which pushes on entry and pops in a
# `finally`, so the stack unwinds even if the body throws.
# =====================================================================
effective_scope() = isempty(session().scope) ? (:default,) : Tuple(session().scope)

pushnamespace!(n)  = (push!(session().scope, Symbol(n)); nothing)
popnamespace!()    = (isempty(session().scope) && error("namespace stack empty"); pop!(session().scope); nothing)
clearnamespace!()  = (empty!(session().scope); nothing)
namespace(f, n)    = (pushnamespace!(n); try f() finally popnamespace!() end)

# register! / lookup implement the (role, scope..., leaf) path rule above.
function register!(s::Session, t::Tensor, name)
    role = Symbol(lowercase(string(roleof(t))), 's')
    anon = get!(() -> Ref(0), s.anon, role)
    leaf = name === nothing ? Symbol('_', anon[] += 1) : Symbol(name)
    path = (role, effective_scope()..., leaf)
    haskey(s.names, path) && error("name already bound: \"", join(path, "/"), "\"")
    s.names[path] = t
    return t
end

function lookup(path::Tuple{Vararg{Symbol}})
    s = session()
    haskey(s.names, path) || error("no tensor named \"", join(path, "/"), "\"")
    return s.names[path]
end

# =====================================================================
# IR builders. These take the Session explicitly; the public Tensor
# constructors pass `session()`. Each forwards `name` to `register!`.
# =====================================================================

# Julia shape (Int | symbolic Dim) -> MLIR ranked tensor type. Symbolic dims
# collapse to MLIR's anonymous `?` (dim_to_mlir); their identity lives only in
# the Dim algebra, not in the IR.
mlir_type(::Type{T}, shape) where {T} =
    IR.TensorType(Int[dim_to_mlir(d) for d in shape], IR.Type(T))

function push_arg!(::Type{T}, shape::NTuple{N,Dim}; name=nothing) where {T,N}
    for d in shape                              # a negative Int would mint invalid MLIR
        d isa Int && d < 0 &&
            error("invalid dimension $d: concrete dims must be ≥ 0 (use a Symbol for a runtime dim)")
    end
    s = session()
    value = IR.push_argument!(s.block, mlir_type(T, shape))
    t = Tensor{T,N,Argument}(value, shape, nothing)
    push!(s.argvars, t)
    register!(s, t, name)
    return t
end

function push_var!(init::Array{T,N}; name=nothing) where {T,N}
    s = session()
    shape = size(init)
    value = IR.push_argument!(s.block, mlir_type(T, shape))
    t = Tensor{T,N,Variable}(value, shape, DataBox{T,N}(init))
    push!(s.argvars, t)
    register!(s, t, name)
    return t
end

# The 1-arg DenseElementsAttribute dispatches per element type to the correct
# raw buffer C call AND permutes column-major -> row-major. (The 2-arg
# shaped-type form routes to the ArrayRef<Attribute> overload and segfaults on
# raw numeric data.)
function push_constant!(data::Array{T,N}; name=nothing) where {T,N}
    s = session()
    op = shlo.constant(; value=IR.DenseElementsAttribute(data), output=mlir_type(T, size(data)))
    push!(s.block, op)
    t = Tensor{T,N,Constant}(IR.result(op, 1), size(data), nothing)
    register!(s, t, name)
    return t
end

# A tensor belongs to this session iff its SSA value lives in the session's
# block — as a block argument (Arg/Var) or as a result of an op in the block
# (Const/Result). Mixing sessions would silently build cross-context IR that
# only fails much later, inside MLIR verification.
function _owned(s::Session, t::Tensor)
    v = t.value
    IR.is_block_arg(v) && return IR.block_owner(v) == s.block
    return IR.block(IR.op_owner(v)) == s.block
end

# Append a finished OpNode's StableHLO op to the block, cross-check the shape
# JOLT computed against the type StableHLO inferred (they can only genuinely
# disagree when BOTH are concrete — `?` matches any symbolic dim), wrap the
# result, and record the node on the tape for the gradient driver. Op results
# are always anonymous. All validation runs BEFORE the op is appended, so a
# rejected op leaves the session's IR untouched.
function push_op!(n::OpNode)
    s = session()
    for (i, t) in enumerate(n.inputs)
        _owned(s, t) ||
            error("input $i of $(typeof(n.op)) belongs to a different session; " *
                  "tensors can only be combined within the session that created them")
    end
    IR.nresults(n.ir) == 1 ||
        error("push_op! expects a single-result op; $(typeof(n.op)) produced $(IR.nresults(n.ir))")
    value = IR.result(n.ir, 1)
    type  = IR.type(value)
    T = IR.julia_type(IR.eltype(type))
    N = length(n.shape)
    IR.ndims(type) == N ||
        error("shape rule bug in $(typeof(n.op)): JOLT computed rank $N, StableHLO inferred $(IR.ndims(type))")
    for i in 1:N
        mlir_ok(n.shape[i], IR.size(type, i)) ||
            error("shape rule bug in $(typeof(n.op)): JOLT computed dim $i = $(n.shape[i]), " *
                  "StableHLO inferred $(Int(IR.size(type, i)))")
    end
    push!(s.block, n.ir)
    t = Tensor{T,N,Result}(value, Tuple(n.shape), nothing)
    push!(s.tape, (n, t))
    register!(s, t, nothing)
    return t
end
