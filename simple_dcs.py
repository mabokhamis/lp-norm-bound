import math
from collections import defaultdict
from itertools import combinations
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, value

# A variable has a name, lower bound, and upper bound.
class Variable:
    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

# A constraint has a name, lower bound, upper bound, and a sum of variables. The sum is
# represented as a dictionary from variable names to coefficients.
class Constraint:
    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.sum = defaultdict(float)

# The objective has a boolean flag to indicate whether it is a maximization or minimization
# and a sum indicating the objective function. The sum is represented as a dictionary from
# variables names to coefficients.
class Objective:
    def __init__(self, maximize):
        self.maximize = maximize
        self.sum = defaultdict(float)

# A linear program has variables, constraints, and an objective.
class LP:
    def __init__(self, maximize):
        self.variables = {}
        self.constraints = {}
        self.objective = Objective(maximize)

    # Add a variable to the LP
    def add_variable(self, name, lower_bound, upper_bound):
        assert name not in self.variables
        self.variables[name] = Variable(name, lower_bound, upper_bound)

    # Add a constraint to the LP
    def add_constraint(self, name, lower_bound, upper_bound):
        assert name not in self.constraints
        self.constraints[name] = Constraint(name, lower_bound, upper_bound)

    # Add `coefficient * variable` to the given constraint
    def add_to_constraint(self, constraint, variable, coefficient):
        assert constraint in self.constraints
        assert variable in self.variables
        self.constraints[constraint].sum[variable] += coefficient

    # Add `coefficient * variable` to the objective
    def add_to_objective(self, variable, coefficient):
        assert variable in self.variables
        self.objective.sum[variable] += coefficient

# Convert an LP to a Pulp LpProblem
def to_lp_problem(lp):
    model = LpProblem("LP", LpMaximize if lp.objective.maximize else LpMinimize)
    # `var_map` is a dictionary from variable names to Pulp LpVariables
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

    model += lpSum(coef * var_map[v] for v, coef in lp.objective.sum.items())

    return model, var_map

# Get a dictionary from variable names to their values
def get_values(var_map):
    return {name: value(v) for name, v in var_map.items()}

###########################################################################################

# A Testcase for the LP conversion
def test_lp1():
    lp = LP(True)
    lp.add_variable("x", 0, float('inf'))
    lp.add_variable("y", 0, float('inf'))
    lp.add_variable("z", 0, float('inf'))

    # C0: x + y <= 1
    lp.add_constraint("C0", float('-inf'), 1)
    lp.add_to_constraint("C0", "x", 1)
    lp.add_to_constraint("C0", "y", 1)

    # C1: y + z <= 1
    lp.add_constraint("C1", float('-inf'), 1)
    lp.add_to_constraint("C1", "y", 1)
    lp.add_to_constraint("C1", "z", 1)

    # C2: x + z <= 1
    lp.add_constraint("C2", float('-inf'), 1)
    lp.add_to_constraint("C2", "x", 1)
    lp.add_to_constraint("C2", "z", 1)

    # Objective: Maximize x + y + z
    lp.add_to_objective("x", 1.0)
    lp.add_to_objective("y", 1.0)
    lp.add_to_objective("z", 1.0)

    model, var_map = to_lp_problem(lp)
    model.solve()

    assert model.status == 1  # 1 corresponds to Optimal
    # The optimal value is 1.5
    assert math.isclose(model.objective.value(), 1.5)

    # The optimal solution is x = 0.5, y = 0.5, z = 0.5
    values = get_values(var_map)
    assert math.isclose(values["x"], 0.5)
    assert math.isclose(values["y"], 0.5)
    assert math.isclose(values["z"], 0.5)

test_lp1()

###########################################################################################

# `DC` is a representation of a bound on an Lp-norm of a degree sequence, as described in
# [this paper](https://arxiv.org/abs/2306.14075). The fields are as follows:
#  - `X` is the set of variables
#  - `Y` is another set of variables
#  - `p` is a number (same "p" from "Lp-norm")
#  - `b` is a real number representing the bound
# Let `deg(Y|X)` be the degree sequence of the variables in `Y` conditioned on the variables
# in `X`. Then, the bound is
# ```
# ||deg(Y|X)||_p <= b`
# ```
# NOTE: A DC is called "simple" if `|X| <= 1`
class DC:
    def __init__(self, X, Y, p, b):
        self.X = set(X)
        self.Y = set(X).union(set(Y)) # Note that we always include X in Y
        self.p = float(p)
        self.b = float(b)

# Given a set `X`, sort `X`, convert each element to a string, and concatenate the strings.
def _name(X):
    return ''.join([str(num) for num in sorted(X)])

# Below, we implement the LP for the Lp-norm bound with *simple* degree constraints, as
# described in [this paper](https://arxiv.org/abs/2211.08381). The LP consists of `n`
# different network flows, where `n` is the number of variables. In particular, for every
# `t ∈ [n]`, there is a different network flow. We refer to it as the "t-flow".

# The flow variable `f_t_X_Y` represents the flow from `X` to `Y` in the t-flow
def flow_var_name(t, X, Y):
    return f"f{t}_{_name(X)}_{_name(Y)}"

# Th capacity constraint `c_t_X_Y` enforces a capacity on the flow from `X` to `Y` in the
# t-flow
def flow_capacity_name(t, X, Y):
    return f"c{t}_{_name(X)}_{_name(Y)}"

# The flow conservation constraint `e_t_Z` enforces flow conservation at `Z` in the t-flow
def flow_conservation_name(t, Z):
    return f"e{t}_{_name(Z)}"

# The DC coefficient `p_i` is the coefficient of the i-th DC in the objective
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
        assert len(dc.X) <= 1, "Only simple degree constraints are supported"
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
    model, var_map = to_lp_problem(lp)
    print(model)
    model.solve()
    assert model.status == 1  # 1 corresponds to Optimal
    values = get_values(var_map)
    for x, v in values.items():
        print(f"{x} = {v}")
    return model.objective.value()

###########################################################################################

# Testcases for the simple_dc_bound function

def test_simple_dc_bound1():
    dcs = [
        DC([], ['A', 'B'], 1, 1),
        DC([], ['A', 'C'], 1, 1),
        DC([], ['B', 'C'], 1, 1),
    ]
    vars = ['A', 'B', 'C']
    assert math.isclose(simple_dc_bound(dcs, vars), 1.5)

# -----------------------------

def test_simple_dc_bound2():
    dcs = [
        DC([], ['A', 'B'], float('inf'), 1),
        DC([], ['A', 'C'], float('inf'), 1),
        DC([], ['B', 'C'], float('inf'), 1),
    ]
    vars = ['A', 'B', 'C']
    assert math.isclose(simple_dc_bound(dcs, vars), 1.5)

# -----------------------------

def test_simple_dc_bound3():
    dcs = [
        DC(['A'], ['B'], 2, 1),
        DC(['B'], ['C'], 2, 1),
        DC(['C'], ['A'], 2, 1),
    ]
    vars = ['A', 'B', 'C']
    assert math.isclose(simple_dc_bound(dcs, vars), 2.0, rel_tol=1e-07)

# -----------------------------

test_simple_dc_bound1()
test_simple_dc_bound2()
test_simple_dc_bound3()
