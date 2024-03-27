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
# log_2 ||deg(Y|X)||_p <= b`
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
# different network flows, where `n` is the number of target variables, i.e. variables whose
# cardinality we are trying to bound. In particular, for every `t ∈ [n]`, there is a
# different network flow. We refer to it as the "t-flow".

# The flow variable `ft_X_Y` represents the flow from `X` to `Y` in the t-flow
def flow_var_name(t, X, Y):
    return f"f{t}_{_name(X)}_{_name(Y)}"

# Th capacity constraint `ct_X_Y` enforces a capacity on the flow from `X` to `Y` in the
# t-flow
def flow_capacity_name(t, X, Y):
    return f"c{t}_{_name(X)}_{_name(Y)}"

# The flow conservation constraint `et_Z` enforces flow conservation at `Z` in the t-flow
def flow_conservation_name(t, Z):
    return f"e{t}_{_name(Z)}"

# The DC coefficient `a_i` is the coefficient of the i-th DC in the objective
def dc_coefficient_name(i):
    return f"a_{i}"

# Given a list of DCs `dcs` and a list of target variables `vars`, construct the vertices
# and edges of the network flow. All t-flows share the same network structure: They only
# differ in flow values.
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

    # For every DC, add the following edges to the network flow:
    #  - One edge from X to Y
    #  - One edge from {} to X
    for dc in dcs:
        _add_edge(dc.X, dc.Y)
        if dc.p != float('inf'):
            _add_edge(set(), dc.X)

    # For every DC, add the following edges to the network flow:
    #  - One edge from Y to {}
    #  - One edge from Y to {y} for every y in Y
    for dc in dcs:
        _add_edge(dc.Y, set())
        for y in dc.Y:
            _add_edge(dc.Y, set({y}))

    return (vertices, edges)

def add_flow_constraints(lp, dcs, vars, vertices, edges):
    for i, dc in enumerate(dcs):
        assert len(dc.X) <= 1, "Only simple degree constraints are supported"
        lp.add_variable(dc_coefficient_name(i), 0.0, float('inf'))

    # For every t-flow
    for t, v in enumerate(vars):
        # For every vertex `Z`, add a flow conservation constraint that enforces the total
        # flow to `Z` to be:
        #   - `+1` if `Z` is the target variable `t`
        #   - `-1` if `Z` is the source {}
        #   - `0` otherwise
        for Z in vertices:
            et_Z = flow_conservation_name(t, Z)
            nt_Z = 1.0 if Z == {v} else -1.0 if not Z else 0.0
            lp.add_constraint(et_Z, nt_Z, float('inf'))

        # For every edge `X -> Y`
        for X, Y in edges:
            # `ft_X_Y` is the t-flow from `X` to `Y`:
            #   - If `X` is a subset of `Y`, then `ft_X_Y` could be anywhere from -inf to
            #     the capacity. (The capacity constraint `ct_X_Y` will be added later)
            #   - Otherwise (i.e. if `Y` is a subset of `X`), then `ft_X_Y` can be
            #     anywhere from 0 to inf
            ft_X_Y = flow_var_name(t, X, Y)
            lp.add_variable(ft_X_Y, -float('inf') if X <= Y else 0.0, float('inf'))
            et_X = flow_conservation_name(t, X)
            et_Y = flow_conservation_name(t, Y)
            # Add `+ft_X_Y` to the flow conservation at `Y` and `-ft_X_Y` to the flow
            # conservation at `X`
            lp.add_to_constraint(et_X, ft_X_Y, -1.0)
            lp.add_to_constraint(et_Y, ft_X_Y, 1.0)
            # If `X` is a subset of `Y`, then there is a capacity constraint on the flow
            # from `X` to `Y`
            if X <= Y:
                ct_X_Y = flow_capacity_name(t, X, Y)
                lp.add_constraint(ct_X_Y, 0.0, float('inf'))
                lp.add_to_constraint(ct_X_Y, ft_X_Y, -1.0)

        # The i-th DC `(X, Y, p, b)` contributes to two capacity constraints:
        #   - It contributes `a_i` to the capacity constraint from `X` to `Y` (where `a_i`
        #     is the coefficient of the i-th DC in the objective)
        #   - It contributes `a_i / p` to the capacity constraint from {} to `X`
        for i, dc in enumerate(dcs):
            a_i = dc_coefficient_name(i)
            if dc.X != dc.Y:
                ct_X_Y = flow_capacity_name(t, dc.X, dc.Y)
                lp.add_to_constraint(ct_X_Y, a_i, 1.0)
            if dc.p != float('inf') and dc.X:
                ct__X = flow_capacity_name(t, set(), dc.X)
                lp.add_to_constraint(ct__X, a_i, 1.0 / dc.p)

# The objective is to minimize the sum of `a_i * dc[i].b` over all DCs
def set_objective(lp, dcs):
    for i, dc in enumerate(dcs):
        a_i = dc_coefficient_name(i)
        lp.add_to_objective(a_i, dc.b)

# Given a list of *simple* DCs `dcs` and a list of target variables `vars`, compute the
# Lp-norm bound on the target variables.
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

def test_simple_dc_bound_JOB_Q1():
    # ||deg({'0MC', '1'}|{'1'})||_1.0 = 1334883.0
    # ||deg({'0MC', '1'}|{'1'})||_2.0 = 1685.8359943956589
    # ||deg({'0MC', '1'}|{'1'})||_3.0 = 232.70072156462575
    # ...
    # ||deg({'1', '0MI_IDX'}|{'1'})||_14.0 = 1.4834790899372283
    # ||deg({'1', '0MI_IDX'}|{'1'})||_15.0 = 1.4449827655296232
    # ||deg({'1', '0MI_IDX'}|{'1'})||_inf = 1.0

    dcs = [
        DC(['1'], ['0MC', '1'], 1.0, math.log2(1334883.0)),
        DC(['1'], ['0MC', '1'], 2.0, math.log2(1685.8359943956589)),
        DC(['1'], ['0MC', '1'], 3.0, math.log2(232.70072156462575)),
        DC(['1'], ['0MC', '1'], 4.0, math.log2(111.6218174166884)),
        DC(['1'], ['0MC', '1'], 5.0, math.log2(89.39599809855387)),
        DC(['1'], ['0MC', '1'], 6.0, math.log2(85.15089958750488)),
        DC(['1'], ['0MC', '1'], 7.0, math.log2(84.28626028158547)),
        DC(['1'], ['0MC', '1'], 8.0, math.log2(84.08192964838128)),
        DC(['1'], ['0MC', '1'], 9.0, math.log2(84.02614781955714)),
        DC(['1'], ['0MC', '1'], 10.0, math.log2(84.00904342807583)),
        DC(['1'], ['0MC', '1'], 11.0, math.log2(84.00331536944712)),
        DC(['1'], ['0MC', '1'], 12.0, math.log2(84.00126786773852)),
        DC(['1'], ['0MC', '1'], 13.0, math.log2(84.00050011506285)),
        DC(['1'], ['0MC', '1'], 14.0, math.log2(84.00020189112921)),
        DC(['1'], ['0MC', '1'], 15.0, math.log2(84.00008295402887)),
        DC(['1'], ['0MC', '1'], float('inf'), math.log2(84.0)),
        DC(['1'], ['1', '0T'], 1.0, math.log2(2528312.0)),
        DC(['1'], ['1', '0T'], 2.0, math.log2(1590.0666652691011)),
        DC(['1'], ['1', '0T'], 3.0, math.log2(136.23129614475658)),
        DC(['1'], ['1', '0T'], 4.0, math.log2(39.87563999823829)),
        DC(['1'], ['1', '0T'], 5.0, math.log2(19.079462387568874)),
        DC(['1'], ['1', '0T'], 6.0, math.log2(11.671816317298546)),
        DC(['1'], ['1', '0T'], 7.0, math.log2(8.216561214186674)),
        DC(['1'], ['1', '0T'], 8.0, math.log2(6.314716145499993)),
        DC(['1'], ['1', '0T'], 9.0, math.log2(5.145476861143866)),
        DC(['1'], ['1', '0T'], 10.0, math.log2(4.368004394179208)),
        DC(['1'], ['1', '0T'], 11.0, math.log2(3.8201068904961626)),
        DC(['1'], ['1', '0T'], 12.0, math.log2(3.4164040038172514)),
        DC(['1'], ['1', '0T'], 13.0, math.log2(3.108317945962265)),
        DC(['1'], ['1', '0T'], 14.0, math.log2(2.8664544674888304)),
        DC(['1'], ['1', '0T'], 15.0, math.log2(2.672116432129139)),
        DC(['1'], ['1', '0T'], 16.0, math.log2(2.5129098960169647)),
        DC(['1'], ['1', '0T'], 17.0, math.log2(2.380329604344474)),
        DC(['1'], ['1', '0T'], 18.0, math.log2(2.2683643581100164)),
        DC(['1'], ['1', '0T'], 19.0, math.log2(2.17265659251006)),
        DC(['1'], ['1', '0T'], 20.0, math.log2(2.089977127668915)),
        DC(['1'], ['1', '0T'], 21.0, math.log2(2.0178863305871544)),
        DC(['1'], ['1', '0T'], 22.0, math.log2(1.9545093733456904)),
        DC(['1'], ['1', '0T'], 23.0, math.log2(1.898383462791728)),
        DC(['1'], ['1', '0T'], 24.0, math.log2(1.848351699168005)),
        DC(['1'], ['1', '0T'], 25.0, math.log2(1.803487874051152)),
        DC(['1'], ['1', '0T'], 26.0, math.log2(1.7630422416840343)),
        DC(['1'], ['1', '0T'], 27.0, math.log2(1.7264017855632618)),
        DC(['1'], ['1', '0T'], 28.0, math.log2(1.6930606803918251)),
        DC(['1'], ['1', '0T'], 29.0, math.log2(1.6625980405997876)),
        DC(['1'], ['1', '0T'], 30.0, math.log2(1.6346609532649696)),
        DC(['1'], ['1', '0T'], float('inf'), math.log2(1.0)),
        DC(['1'], ['1', '0MI_IDX'], 1.0, math.log2(250.0)),
        DC(['1'], ['1', '0MI_IDX'], 2.0, math.log2(15.811388300841896)),
        DC(['1'], ['1', '0MI_IDX'], 3.0, math.log2(6.299605249474365)),
        DC(['1'], ['1', '0MI_IDX'], 4.0, math.log2(3.976353643835253)),
        DC(['1'], ['1', '0MI_IDX'], 5.0, math.log2(3.017088168272582)),
        DC(['1'], ['1', '0MI_IDX'], 6.0, math.log2(2.509901442183411)),
        DC(['1'], ['1', '0MI_IDX'], 7.0, math.log2(2.2007102102809872)),
        DC(['1'], ['1', '0MI_IDX'], 8.0, math.log2(1.9940796483178032)),
        DC(['1'], ['1', '0MI_IDX'], 9.0, math.log2(1.8468761744797573)),
        DC(['1'], ['1', '0MI_IDX'], 10.0, math.log2(1.736976732219687)),
        DC(['1'], ['1', '0MI_IDX'], 11.0, math.log2(1.6519410534528962)),
        DC(['1'], ['1', '0MI_IDX'], 12.0, math.log2(1.584266846899035)),
        DC(['1'], ['1', '0MI_IDX'], 13.0, math.log2(1.5291740650985803)),
        DC(['1'], ['1', '0MI_IDX'], 14.0, math.log2(1.4834790899372283)),
        DC(['1'], ['1', '0MI_IDX'], 15.0, math.log2(1.4449827655296232)),
        DC(['1'], ['1', '0MI_IDX'], float('inf'), math.log2(1.0)),
    ]

    vars = ['0MC', '0MI_IDX', '0T', '1']
    bound = pow(2, simple_dc_bound(dcs, vars))
    print(f"Bound: {bound}")
    assert math.isclose(bound, 7017, abs_tol=1.0)

# -----------------------------

test_simple_dc_bound1()
test_simple_dc_bound2()
test_simple_dc_bound3()
test_simple_dc_bound_JOB_Q1()
