# =====================================================================
# Session
#
# Holds the MLIR context and the block we accumulate ops into, plus the name
# registry. `mod` is filled at build time. `argvars` records Argument/Variable
# tensors in block-argument order (that ordering becomes the compiled graph's
# input signature). A Variable carries its initializer, not a value — its array
# is realized at compile time, so there's no host-side value map to keep in sync.
# =====================================================================
mutable struct Session
    context::IR.Context
    block::IR.Block
    dtype::DataType                             # default element type when a constructor is given none
    argvars::Vector{AbstractTensor}             # Argument/Variable tensors, in block-argument order
    scope::Vector{Symbol}                       # current namespace stack (mutable)
    names::Dict{Tuple{Vararg{Symbol}},AbstractTensor}  # scoped path -> tensor
    anon::Dict{Symbol,Base.RefValue{Int}}             # monotonic counters for anonymous names
    outputs::Vector{AbstractTensor}             # set when the graph is finalized
    mod::Union{Nothing,IR.Module}
    facts::Facts                                # symbolic-dim constraints (dims.jl)
    tape::Vector{Tuple{OpNode,AbstractTensor}}  # (node, output), in emission order — the AD tape
end

function Session()
    ctx   = Reactant.ReactantContext()
    block = IR.Block()
    return Session(
        ctx, block,
        Float32,                                # dtype seed (default element type)
        AbstractTensor[],
        Symbol[],
        Dict{Tuple{Vararg{Symbol}},AbstractTensor}(),
        Dict{Symbol,Base.RefValue{Int}}(),
        AbstractTensor[],
        nothing,
        Facts(),
        Tuple{OpNode,AbstractTensor}[],
    )
end

function Base.show(io::IO, s::Session)
    print(io, "Session(", length(s.argvars), " inputs, ",
          length(s.names), " named, ", length(s.outputs), " outputs)")
end

# =====================================================================
# Global session (transparent: you never have to create one).
#
# Lifecycle — install, retrieve, or replace the one active session:
#   session()          -> the active session (building + activating one on first use)
#   session!(s)        -> install `s` as the active session (activating its context)
#   new_session!()     -> fresh session, installed and returned
#   reset_session!()   -> drop the current session and pop its context
#   with_session(f, s) -> run `f` with `s` active, then restore the previous one
#
# Active-session accessors — reach through the current session so callers never
# touch it directly:
#   current_facts()    -> its symbolic-dim constraint store (dims.jl)
#   default_dtype()    -> the element type a constructor falls back to when given none
#   default_dtype!(T)  -> set that default for the active session
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

# Active-session accessors — each reaches through session() into whatever
# session is current, so the dims/ops/tensor layers stay session-agnostic.

# Its symbolic-dim constraint store: the single point through which dims.jl's
# convenience queries (dims_equal, same_dim!, …) reach the session.
current_facts() = session().facts

# Its default element type — what a tensor constructor uses when given no dtype.
# A Session field (seeded to Float32 in Session()), so a fresh session starts
# from the default; default_dtype!(T) changes it for the active session only.
default_dtype()            = session().dtype
default_dtype!(::Type{T}) where {T<:Real} = (session().dtype = T; T)

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
#       Var(8)                         ->  (:variables, :default, :_1)
#       x + y   (a Result)             ->  (:results,   :default, :_1)
#       namespace("params") do
#           Var(8)                     ->  (:variables, :params,  :_2)
#           Var(8, name="W")           ->  (:variables, :params,  :W)
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

# _resolve! computes the (role, scope..., leaf) path and rejects a duplicate name
# BEFORE any IR mutation — so a rejected name leaves the session untouched, the
# same transactional guarantee push_op! gives for a rejected op. The role is
# known statically (no built tensor, hence no emitted IR, is needed to resolve
# it); the builder commits `s.names[path] = t` only after the IR is built.
function _resolve!(s::Session, ::Type{R}, name) where {R<:AbstractTensor}
    role = Symbol(lowercase(String(nameof(R))), 's')       # Argument -> :arguments
    ctr  = get!(() -> Ref(0), s.anon, role)                # ensure the per-role counter exists
    leaf = name === nothing ? Symbol('_', ctr[] += 1) : Symbol(name)   # anon leaves never collide
    path = (role, effective_scope()..., leaf)
    haskey(s.names, path) && error("name already bound: \"", join(path, "/"), "\"")
    return path
end

function lookup(path::Tuple{Vararg{Symbol}})
    s = session()
    haskey(s.names, path) || error("no tensor named \"", join(path, "/"), "\"")
    return s.names[path]
end

# Fetch a previously-named tensor from the active session's registry. The
# role-specific getters supply the role head implicitly, so you pass the scope
# path + leaf (top-level scope is "default"); `getTensor` takes the full path.
#     getVariable("Dense", "W")   ==  getTensor("variables", "Dense", "W")
#     getArgument("default", "x") ==  getTensor("arguments", "default", "x")
getArgument(path::AbstractString...) = lookup((:arguments, Symbol.(path)...))
getVariable(path::AbstractString...) = lookup((:variables, Symbol.(path)...))
getConstant(path::AbstractString...) = lookup((:constants, Symbol.(path)...))
getTensor(path::AbstractString...)   = lookup(Symbol.(path))
const getArg   = getArgument
const getVar   = getVariable
const getConst = getConstant

# =====================================================================
# IR builders. These take the Session explicitly; the public constructors
# (Argument/Variable/Constant) reach the active session via `session()`. Each
# forwards `name` to the name registry.
# =====================================================================

# Julia shape (Int | symbolic Dim) -> MLIR ranked tensor type. Symbolic dims
# collapse to MLIR's anonymous `?` (dim_to_mlir); their identity lives only in
# the Dim algebra, not in the IR.
# REVERSE-DIMS: MLIR/StableHLO/IREE are row-major; Julia is column-major. A
# Julia column-major (m,n) buffer is byte-identical to a row-major (n,m) tensor,
# so we build every MLIR type with the shape REVERSED. This is what lets the
# host↔IREE boundary be copy-free (no transpose): the whole graph lives in
# reversed-dim space, and the shape rules (Julia order) reverse only here at the
# lowering edge. Ops that name a dimension remap Julia axis i → MLIR axis N-i.
mlir_type(::Type{T}, shape) where {T} =
    IR.TensorType(Int[dim_to_mlir(d) for d in reverse(shape)], IR.Type(T))

function push_arg!(shape::NTuple{N,Dim}, ::Type{T}=default_dtype(); name=nothing) where {N,T}
    for d in shape                              # a negative Int would mint invalid MLIR
        d isa Int && d < 0 &&
            error("invalid dimension $d: concrete dims must be ≥ 0 (use a Symbol for a runtime dim)")
    end
    s = session()
    path  = _resolve!(s, Argument, name)          # collision check BEFORE mutating the block
    value = IR.push_argument!(s.block, mlir_type(T, shape))
    t = Argument{T,N}(value, shape)
    push!(s.argvars, t)
    s.names[path] = t
    return t
end

# A Variable is stored as shape + initializer; its array is realized at compile
# time (the ops/compile layer calls `init(rng, T, dims...)`), so nothing is
# allocated here. `init` must be a function with the (rng, T, dims...) signature.
function push_var!(init, dims::NTuple{N,Int}, ::Type{T}=default_dtype(); name=nothing) where {N,T}
    init isa Function ||
        error("a Variable's init must be a function; got $(typeof(init))")
    # Verify the signature WITHOUT calling it (no probe allocation): does a method
    # exist for a rank-N call init(rng::AbstractRNG, T::Type, dims::Int...)? Catches
    # wrong arity, a missing T argument (e.g. Base.ones), or `Zeros` without `()`.
    hasmethod(init, Tuple{AbstractRNG,DataType,ntuple(_ -> Int, N)...}) ||
        error("init has the wrong signature for a rank-$N Variable — it must be callable as " *
              "init(rng::AbstractRNG, T::Type, dims::Integer...) -> Array (note the T argument). " *
              "Use Zeros()/Ones()/Fill(…)/RandN(…)/… (with parentheses) or your own closure.")
    for d in dims                               # a negative Int would mint invalid MLIR
        d < 0 && error("invalid dimension $d: variable dims must be ≥ 0")
    end
    s = session()
    path  = _resolve!(s, Variable, name)          # collision check BEFORE mutating the block
    value = IR.push_argument!(s.block, mlir_type(T, dims))
    t = Variable{T,N}(value, dims, init)
    push!(s.argvars, t)
    s.names[path] = t
    return t
end

# The 1-arg DenseElementsAttribute dispatches per element type to the correct
# raw buffer C call AND permutes column-major -> row-major. (The 2-arg
# shaped-type form routes to the ArrayRef<Attribute> overload and segfaults on
# raw numeric data.)
function push_constant!(data::Array{T,N}; name=nothing) where {T,N}
    s = session()
    path = _resolve!(s, Constant, name)           # collision check BEFORE mutating the block
    op = shlo.constant(; value=IR.DenseElementsAttribute(data), output=mlir_type(T, size(data)))
    push!(s.block, op)
    t = Constant{T,N}(IR.result(op, 1), size(data), data)   # keep the value for getvalue
    s.names[path] = t
    return t
end

# A tensor belongs to this session iff its SSA value lives in the session's
# block — as a block argument (Arg/Var) or as a result of an op in the block
# (Const/Result). Mixing sessions would silently build cross-context IR that
# only fails much later, inside MLIR verification.
function _owned(s::Session, t::AbstractTensor)
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
    for i in 1:N                                   # MLIR type is reversed (see mlir_type)
        mlir_ok(n.shape[i], IR.size(type, N - i + 1)) ||
            error("shape rule bug in $(typeof(n.op)): JOLT computed dim $i = $(n.shape[i]), " *
                  "StableHLO inferred $(Int(IR.size(type, N - i + 1)))")
    end
    path = _resolve!(s, Result, nothing)          # results never collide; resolve, then append
    push!(s.block, n.ir)
    t = Result{T,N}(value, Tuple(n.shape))
    push!(s.tape, (n, t))
    s.names[path] = t
    return t
end
