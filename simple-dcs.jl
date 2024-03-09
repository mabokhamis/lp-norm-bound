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

function add_constraint!(lp::LP, name::Symbol, lower_bound::Float64, upper_bound::Float64)
    @assert !haskey(lp.constraints, name)
    return lp.constraints[name] = Constraint(name, lower_bound, upper_bound, Sum())
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
    @assert haskey(lp.variables, variable)
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

# lp = LP(true)
# add_variable!(lp, :x, 0.0, 1.0)
# add_variable!(lp, :y, 0.0, Inf)
# add_variable!(lp, :z, 0.0, Inf)
# add_constraint!(lp, :xy, -Inf, 1.0)
# add_to_constraint!(lp, :xy, :x, 1.0)
# add_to_constraint!(lp, :xy, :y, 1.0)
# add_constraint!(lp, :yz, -Inf, 1.0)
# add_to_constraint!(lp, :yz, :y, 1.0)
# add_to_constraint!(lp, :yz, :z, 1.0)
# add_constraint!(lp, :xz, -Inf, 1.0)
# add_to_constraint!(lp, :xz, :x, 1.0)
# add_to_constraint!(lp, :xz, :z, 1.0)
# add_to_objective!(lp, :x, 1.0)
# add_to_objective!(lp, :y, 1.0)
# add_to_objective!(lp, :z, 1.0)
# model = to_jump(lp)
# println(model)
# optimize!(model)
# @assert termination_status(model) == MathOptInterface.OPTIMAL
# println(objective_value(model))


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
    return Symbol("f$(t)_$(_name(X))_$(_name(Y))")
end

function flow_capacity_name(t::Int, X::Set{T}, Y::Set{T}) where T
    return Symbol("c$(t)_$(_name(X))_$(_name(Y))")
end

function flow_conservation_name(t::Int, Z::Set{T}) where T
    return Symbol("e$(t)_$(_name(Z))")
end

function dc_coefficient_name(i::Int)
    return Symbol("p_$i")
end

function _collect_vertices_and_edges(dcs::Vector{DC{T}}, vars::Vector{T}) where T
    vertices = Set{Set{T}}()
    edges = Set{Tuple{Set{T},Set{T}}}()

    function _add_edge(X::Set{T}, Y::Set{T})
        @assert X ⊆ Y || Y ⊆ X
        push!(vertices, X)
        push!(vertices, Y)
        X != Y && (Y, X) ∉ edges && push!(edges, (X, Y))
    end

    push!(vertices, Set{T}())
    for v ∈ vars
        push!(vertices, Set{T}((v,)))
    end

    for dc ∈ dcs
        _add_edge(dc.X, dc.Y)
        !isinf(dc.p) && _add_edge(Set{T}(), dc.X)
    end

    for dc ∈ dcs
        _add_edge(dc.Y, Set{T}())
        for y ∈ dc.Y
            _add_edge(dc.Y, Set{T}((y,)))
        end
    end

    return (vertices, edges)
end

function add_flow_constraints!(
    lp::LP,
    dcs::Vector{DC{T}},
    vars::Vector{T},
    vertices::Set{Set{T}},
    edges::Set{Tuple{Set{T},Set{T}}}
) where T
    for i ∈ eachindex(dcs)
        add_variable!(lp, dc_coefficient_name(i), 0.0, Inf)
    end
    for (t, v) ∈ enumerate(vars)
        for Z ∈ vertices
            e_Z = flow_conservation_name(t, Z)
            @assert Z isa Set{T}
            n_Z = if Z == Set{T}((v,))
                1.0
            elseif isempty(Z)
                -1.0
            else
                0.0
            end
            add_constraint!(lp, e_Z, n_Z, Inf)
        end
        for (X, Y) ∈ edges
            f_X_Y = flow_var_name(t, X, Y)
            add_variable!(lp, f_X_Y, -Inf, Inf)
            e_X = flow_conservation_name(t, X)
            e_Y = flow_conservation_name(t, Y)
            add_to_constraint!(lp, e_X, f_X_Y, -1.0)
            add_to_constraint!(lp, e_Y, f_X_Y, +1.0)
            if X ⊆ Y
                c_X_Y = flow_capacity_name(t, X, Y)
                add_constraint!(lp, c_X_Y, 0.0, Inf)
                add_to_constraint!(lp, c_X_Y, f_X_Y, -1.0)
            end
        end
        for (i, dc) ∈ enumerate(dcs)
            p_i = dc_coefficient_name(i)
            c_X_Y = flow_capacity_name(t, dc.X, dc.Y)
            add_to_constraint!(lp, c_X_Y, p_i, 1.0)
            if !isinf(dc.p)
                c__X = flow_capacity_name(t, Set{T}(), dc.X)
                add_to_constraint!(lp, c__X, p_i, 1.0 / dc.p)
            end
        end
    end
end

function set_objective!(lp::LP, dcs::Vector{DC{T}}) where T
    for (i, dc) ∈ enumerate(dcs)
        p_i = dc_coefficient_name(i)
        add_to_objective!(lp, p_i, dc.b)
    end
end

function simple_dc_bound(dcs::Vector{DC{T}}, vars::Vector{T}) where T
    lp = LP(false)
    (vertices, edges) = _collect_vertices_and_edges(dcs, vars)
    add_flow_constraints!(lp, dcs, vars, vertices, edges)
    set_objective!(lp, dcs)
    model = to_jump(lp)
    optimize!(model)
    @assert termination_status(model) == MathOptInterface.OPTIMAL
    return objective_value(model)
end

###########################################################################################

dcs = [
    DC(Symbol[], [:A, :B], Inf, 1),
    DC(Symbol[], [:A, :C], Inf, 1),
    DC(Symbol[], [:B, :C], Inf, 1),
]

vars = [:A, :B, :C]

println(simple_dc_bound(dcs, vars))

end
