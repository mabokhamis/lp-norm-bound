from collections import defaultdict
from itertools import combinations

from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, value

class Variable:
    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

class Constraint:
    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.sum = defaultdict(float)

class Objective:
    def __init__(self, maximize):
        self.maximize = maximize
        self.sum = defaultdict(float)

class LP:
    def __init__(self, maximize):
        self.variables = {}
        self.constraints = {}
        self.objective = Objective(maximize)

    def add_variable(self, name, lower_bound, upper_bound):
        assert name not in self.variables
        self.variables[name] = Variable(name, lower_bound, upper_bound)

    def add_constraint(self, name, lower_bound, upper_bound):
        assert name not in self.constraints
        self.constraints[name] = Constraint(name, lower_bound, upper_bound)

    def add_to_constraint(self, constraint, variable, coefficient):
        assert constraint in self.constraints
        assert variable in self.variables
        self.constraints[constraint].sum[variable] += coefficient

    def add_to_objective(self, variable, coefficient):
        assert variable in self.variables
        self.objective.sum[variable] += coefficient

def to_jump(lp):
    model = LpProblem("LP", LpMaximize if lp.objective.maximize else LpMinimize)
    var_map = {name: LpVariable(
            name,
            lowBound = var.lower_bound if var.lower_bound != float('-inf') else None,
            upBound  = var.upper_bound if var.upper_bound != float('inf')  else None,
        )
        for name, var in lp.variables.items()}

    for con in lp.constraints.values():
        if con.lower_bound == con.upper_bound:
            model += lpSum(coef * var_map[v] for v, coef in con.sum.items()) == con.lower_bound
        else:
            if con.upper_bound != float('inf'):
                model += lpSum(coef * var_map[v] for v, coef in con.sum.items()) <= con.upper_bound
            if con.lower_bound != float('-inf'):
                model += lpSum(coef * var_map[v] for v, coef in con.sum.items()) >= con.lower_bound

    if lp.objective.maximize:
        model += lpSum(coef * var_map[v] for v, coef in lp.objective.sum.items())
    else:
        model += -lpSum(coef * var_map[v] for v, coef in lp.objective.sum.items())

    return model, var_map

def get_values(var_map):
    return {name: value(v) for name, v in var_map.items()}

###########################################################################################

# def main():
#     lp = LP(True)
#     lp.add_variable("x", 0, 1.0)
#     lp.add_variable("y", 0, float('inf'))
#     lp.add_variable("z", 0, float('inf'))
#     lp.add_constraint("xy", 0, 1)
#     lp.add_to_constraint("xy", "x", 1.0)
#     lp.add_to_constraint("xy", "x", 1.0)
#     lp.add_to_constraint("xy", "y", 1.0)
#     lp.add_constraint("yz", float('-inf'), 1.0)
#     lp.add_to_constraint("yz", "y", 1.0)
#     lp.add_to_constraint("yz", "z", 1.0)
#     lp.add_constraint("xz", float('-inf'), 1.0)
#     lp.add_to_constraint("xz", "x", 1.0)
#     lp.add_to_constraint("xz", "z", 1.0)
#     lp.add_to_objective("x", 1.0)
#     lp.add_to_objective("y", 1.0)
#     lp.add_to_objective("z", 1.0)

#     model, var_map = to_jump(lp)
#     print(model)
#     model.solve()

#     assert model.status == 1  # 1 corresponds to Optimal
#     print("Objective value:", model.objective.value())
#     for name, var in var_map.items():
#         print(name, "=", var.varValue)

# if __name__ == "__main__":
#     main()

###########################################################################################

class DC:
    def __init__(self, X, Y, p, b):
        self.X = set(X)
        self.Y = set(Y)
        self.p = float(p)
        self.b = float(b)

def _name(X):
    return ''.join([str(num) for num in sorted(X)])

def flow_var_name(t, X, Y):
    return f"f{t}_{_name(X)}_{_name(Y)}"

def flow_capacity_name(t, X, Y):
    return f"c{t}_{_name(X)}_{_name(Y)}"

def flow_conservation_name(t, Z):
    return f"e{t}_{_name(Z)}"

def dc_coefficient_name(i):
    return f"p_{i}"

def _collect_vertices_and_edges(dcs, vars):
    vertices = set()
    edges = set()

    def _add_edge(X, Y):
        assert X <= Y or Y <= X
        X = frozenset(X)
        Y = frozenset(Y)
        vertices.add(X)
        vertices.add(Y)
        if X != Y and (Y, X) not in edges:
            edges.add((X, Y))

    vertices.add(frozenset())
    for v in vars:
        vertices.add(frozenset({v}))

    for dc in dcs:
        _add_edge(dc.X, dc.Y)
        if dc.p != float('inf'):
            _add_edge(set(), dc.X)

    for dc in dcs:
        _add_edge(dc.Y, set())
        for y in dc.Y:
            _add_edge(dc.Y, set({y}))

    return (vertices, edges)

def add_flow_constraints(lp, dcs, vars, vertices, edges):
    for i, dc in enumerate(dcs):
        lp.add_variable(dc_coefficient_name(i), 0.0, float('inf'))

    for t, v in enumerate(vars):
        for Z in vertices:
            e_Z = flow_conservation_name(t, Z)
            n_Z = 1.0 if Z == {v} else -1.0 if not Z else 0.0
            lp.add_constraint(e_Z, n_Z, float('inf'))

        for X, Y in edges:
            f_X_Y = flow_var_name(t, X, Y)
            lp.add_variable(f_X_Y, -float('inf') if X <= Y else 0.0, float('inf'))
            e_X = flow_conservation_name(t, X)
            e_Y = flow_conservation_name(t, Y)
            lp.add_to_constraint(e_X, f_X_Y, -1.0)
            lp.add_to_constraint(e_Y, f_X_Y, 1.0)
            if X <= Y:
                c_X_Y = flow_capacity_name(t, X, Y)
                lp.add_constraint(c_X_Y, 0.0, float('inf'))
                lp.add_to_constraint(c_X_Y, f_X_Y, -1.0)

        for i, dc in enumerate(dcs):
            p_i = dc_coefficient_name(i)
            if dc.X != dc.Y:
                c_X_Y = flow_capacity_name(t, dc.X, dc.Y)
                lp.add_to_constraint(c_X_Y, p_i, 1.0)
            if dc.p != float('inf') and dc.X:
                c__X = flow_capacity_name(t, set(), dc.X)
                lp.add_to_constraint(c__X, p_i, 1.0 / dc.p)

def set_objective(lp, dcs):
    for i, dc in enumerate(dcs):
        p_i = dc_coefficient_name(i)
        lp.add_to_objective(p_i, dc.b)

def simple_dc_bound(dcs, vars):
    lp = LP(False)
    vertices, edges = _collect_vertices_and_edges(dcs, vars)
    add_flow_constraints(lp, dcs, vars, vertices, edges)
    set_objective(lp, dcs)
    model, var_map = to_jump(lp)
    print(model)
    model.solve()
    assert model.status == 1  # 1 corresponds to Optimal
    values = get_values(var_map)
    for x, v in values.items():
        print(f"{x} = {v}")
    return model.objective.value()

###########################################################################################

dcs = [
    DC([], ['A', 'B'], 1, 1),
    DC([], ['A', 'C'], 1, 1),
    DC([], ['B', 'C'], 1, 1),
]

vars = ['A', 'B', 'C']

print(simple_dc_bound(dcs, vars))

# dcs = [
#     DC([], ['A', 'B'], float('inf'), 1),
#     DC([], ['A', 'C'], float('inf'), 1),
#     DC([], ['B', 'C'], float('inf'), 1),
# ]

# vars = ['A', 'B', 'C']

# print(simple_dc_bound(dcs, vars))

# dcs = [
#     DC({'A'}, {'B'}, 2, 1),
#     DC({'B'}, {'C'}, 2, 1),
#     DC({'C'}, {'A'}, 2, 1),
# ]

# vars = ['A', 'B', 'C']

# print(simple_dc_bound(dcs, vars))
