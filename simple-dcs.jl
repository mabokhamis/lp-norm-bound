# import Pkg
# Pkg.add("Clp")
# Pkg.add("DataStructures")
# Pkg.add("JuMP")
# Pkg.add("MathOptInterface")

module SimpleDCs

using Clp
using DataStructures
using JuMP
using MathOptInterface

###########################################################################################

const Sum = SortedDict{Symbol,Float64}

struct Variable
    name::Symbol
    lower_bound::Float64
    upper_bound::Float64
end

mutable struct Constraint
    name::Symbol
    lower_bound::Float64
    upper_bound::Float64
    sum::Sum
end

mutable struct Objective
    maximize::Bool
    sum::Sum
end

mutable struct LP
    variables::SortedDict{Symbol,Variable}
    constraints::SortedDict{Symbol,Constraint}
    objective::Objective
end

function LP(maximize::Bool)
    return LP(
        SortedDict{Symbol,Variable}(),
        SortedDict{Symbol,Constraint}(),
        Objective(maximize, Sum())
    )
end

function add_variable!(lp::LP, name::Symbol, lower_bound::Float64, upper_bound::Float64)
    @assert !haskey(lp.variables, name)
    return lp.variables[name] = Variable(name, lower_bound, upper_bound)
end

function add_constraint!(
    lp::LP, name::Symbol, lower_bound::Float64, upper_bound::Float64, sum::Sum = Sum()
)
    @assert !haskey(lp.constraints, name)
    return lp.constraints[name] = Constraint(name, lower_bound, upper_bound, sum)
end

function add_to_constraint!(
    lp::LP, constraint::Symbol, variable::Symbol, coefficient::Float64
)
    @assert haskey(lp.constraints, constraint)
    @assert haskey(lp.variables, variable)
    if haskey(lp.constraints[constraint].sum, variable)
        lp.constraints[constraint].sum[variable] += coefficient
    else
        lp.constraints[constraint].sum[variable] = coefficient
    end
end

function add_to_objective!(lp::LP, variable::Symbol, coefficient::Float64)
    if haskey(lp.objective.sum, variable)
        lp.objective.sum[variable] += coefficient
    else
        lp.objective.sum[variable] = coefficient
    end
end

function to_jump(model::Model, var::Variable)
    return @variable(
        model;
        base_name = String(var.name),
        lower_bound = var.lower_bound,
        upper_bound = var.upper_bound
    )
end

function to_jump(model::Model, con::Constraint, var_map::Dict{Symbol,VariableRef})
    if con.lower_bound == con.upper_bound
        return @constraint(
            model,
            sum(c * var_map[v] for (v, c) ∈ con.sum) == con.lower_bound
        )
    elseif con.lower_bound != -Inf && con.upper_bound != Inf
        return @constraint(
            model,
            con.lower_bound <= sum(c * var_map[v] for (v, c) ∈ con.sum) <= con.upper_bound
        )
    elseif con.lower_bound != -Inf
        return @constraint(
            model,
            con.lower_bound <= sum(c * var_map[v] for (v, c) ∈ con.sum)
        )
    else
        return @constraint(
            model,
            sum(c * var_map[v] for (v, c) ∈ con.sum) <= con.upper_bound
        )
    end
end

function to_jump(model::Model, obj::Objective, var_map::Dict{Symbol,VariableRef})
    if obj.maximize
        return @objective(
            model,
            Max,
            sum(c * var_map[v] for (v, c) ∈ obj.sum)
        )
    else
        return @objective(
            model,
            Min,
            sum(c * var_map[v] for (v, c) ∈ obj.sum)
        )
    end
end

function to_jump(lp::LP)
    model = Model(Clp.Optimizer)
    set_optimizer_attribute(model, "LogLevel", 0)
    var_map = Dict(name => to_jump(model, var) for (name, var) ∈ lp.variables)
    for (_, con) ∈ lp.constraints
        to_jump(model, con, var_map)
    end
    to_jump(model, lp.objective, var_map)
    return model
end

###########################################################################################

lp = LP(true)
add_variable!(lp, :x, 0.0, 1.0)
add_variable!(lp, :y, 0.0, Inf)
add_variable!(lp, :z, 0.0, Inf)
add_constraint!(lp, :xy, -Inf, 1.0)
add_to_constraint!(lp, :xy, :x, 1.0)
add_to_constraint!(lp, :xy, :y, 1.0)
add_constraint!(lp, :yz, -Inf, 1.0)
add_to_constraint!(lp, :yz, :y, 1.0)
add_to_constraint!(lp, :yz, :z, 1.0)
add_constraint!(lp, :xz, -Inf, 1.0)
add_to_constraint!(lp, :xz, :x, 1.0)
add_to_constraint!(lp, :xz, :z, 1.0)
add_to_objective!(lp, :x, 1.0)
add_to_objective!(lp, :y, 1.0)
add_to_objective!(lp, :z, 1.0)
model = to_jump(lp)
println(model)
optimize!(model)
@assert termination_status(model) == MathOptInterface.OPTIMAL
println(objective_value(model))


###########################################################################################

mutable struct DC{T}
    X::Set{T}
    Y::Set{T}
    p::Float64
    b::Float64

    function DC{T}(X::Set{T}, Y::Set{T}, p::Float64, b::Float64) where T
        return new{T}(X, X ∪ Y, p, b)
    end
end

const MySet{T} = Union{AbstractSet{T},AbstractVector{T}}

function DC(X::MySet{T}, Y::MySet{T}, p::Number, b::Number) where T
    return DC{T}(Set{T}(X), Set{T}(Y), Float64(p), Float64(b))
end

function _name(X::Set{T}) where T
    return join(sort!(collect(X)))
end

function flow_var_name(t::Int, X::Set{T}, Y::Set{T}) where T
    return Symbol("f$t_$(_name(X))_$(_name(Y))")
end

function flow_con_name(t::Int, Z::Set{T}) where T
    return Symbol("e$t_$(_name(Z))")
end

function dc_coefficient_name(i::Int)
    return Symbol("p_$i")
end

function simple_dc_bound(dcs::Vector{DC{T}}, vars::Vector{T}) where T
    lp = LP(false)
    return to_jump(lp)
end

###########################################################################################

dcs = [
    DC(Symbol[], [:A], Inf, 1),
    DC([:A], [:B], Inf, 1),
    DC([:B], [:C], Inf, 1),
]

end
