# =====================================================================
# Stateful variables — assign!, Flag, and train/test mode.
#
# A variable is "state" purely by how you USE it: call `assign!(v, newval)` and
# `v`'s slot in `vars` comes back holding `newval` after `fn` runs. Nothing about
# the variable's type, scope, or gradient changes — a variable can be assigned
# AND receive gradients (`wrt=v`); whether it's trained is only about what you
# hand to Optimisers. State advances IN PLACE (the forward writes the new value
# into the shared `vars` arena, zero read-copy); `deepcopy(vars)` to snapshot.
#
# `Flag` is sugar for a 0/1 scalar variable scoped under `variables/flags`, the
# location `trainmode!`/`testmode!` flip. The graph branches on it with `select`.
# =====================================================================

# --- assign!: declare a variable's next value -----------------------
# Reads the OLD value (`newval` is built from the incoming `v`) and records the
# NEW one; last call wins. Not auto-stop_gradient'd — the caller decides whether
# the update is on the gradient path (BN detaches its stats; meta-learning does not).
function assign!(v::Variable, newval::AbstractTensor)
    _samedims("assign!", size(v), size(newval))       # the update must match the variable's shape
    session().assigns[v] = newval
    return newval
end
# Conditional sugar: update only when `flag` is on, else keep the old value.
#   assign!(v, flag, newval)  ==  assign!(v, select(flag != 0, newval, v))
assign!(v::Variable, flag::AbstractTensor, newval::AbstractTensor) =
    assign!(v, select(flag .!= zero(eltype(flag)), newval, v))

# --- Flag: a 0/1 scalar variable under variables/flags --------------
#   Flag()                → an (anonymous) flag, default on (1)
#   Flag(false)           → default off (0);  1/0 accepted too
#   Flag(true, :training) → named `training`  → vars.flags.training
function Flag(v::Union{Bool,Integer} = true, name = nothing)
    fv = convert(default_dtype(), iszero(v) ? 0 : 1)
    return namespace("flags") do
        Variable(default_dtype(), Fill(fv); name = name === nothing ? nothing : String(name))
    end
end

# --- mode switching -------------------------------------------------
# Flip every flag under `vars.flags` in place (no-op if the model declared none).
# Mutating the shared arena is seen by the next `fn` call with no copy.
trainmode!(vars) = (:flags in propertynames(vars) && (vars.flags .= 1); vars)
testmode!(vars)  = (:flags in propertynames(vars) && (vars.flags .= 0); vars)
