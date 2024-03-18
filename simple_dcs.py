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

def main():
    lp = LP(True)
    lp.add_variable("x", 0, 1.0)
    lp.add_variable("y", 0, float('inf'))
    lp.add_variable("z", 0, float('inf'))
    lp.add_constraint("xy", 0, 1)
    lp.add_to_constraint("xy", "x", 1.0)
    lp.add_to_constraint("xy", "x", 1.0)
    lp.add_to_constraint("xy", "y", 1.0)
    lp.add_constraint("yz", float('-inf'), 1.0)
    lp.add_to_constraint("yz", "y", 1.0)
    lp.add_to_constraint("yz", "z", 1.0)
    lp.add_constraint("xz", float('-inf'), 1.0)
    lp.add_to_constraint("xz", "x", 1.0)
    lp.add_to_constraint("xz", "z", 1.0)
    lp.add_to_objective("x", 1.0)
    lp.add_to_objective("y", 1.0)
    lp.add_to_objective("z", 1.0)

    model, var_map = to_jump(lp)
    print(model)
    model.solve()

    assert model.status == 1  # 1 corresponds to Optimal
    print("Objective value:", model.objective.value())
    for name, var in var_map.items():
        print(name, "=", var.varValue)

if __name__ == "__main__":
    main()
