#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <cmath>
#include <numeric>
#include <cassert>
#include "Highs.h"

using namespace std;

// A variable has a name, lower bound, and upper bound.
struct Variable {
    const string name;
    double lower_bound;
    double upper_bound;
};

// A constraint has a name, lower bound, upper bound, and a sum of variables. The sum is
// represented as a dictionary from variable names to coefficients.
struct Constraint {
    const string name;
    double lower_bound;
    double upper_bound;
    unordered_map<int, double> sum;
};

// The objective has a boolean flag to indicate whether it is a maximization or minimization
// and a sum indicating the objective function. The sum is represented as a dictionary from
// variables names to coefficients.
struct Objective {
    bool maximize;
    unordered_map<int, double> sum;

    Objective(bool maximize_) : maximize(maximize_) {}
};

// A linear program has variables, constraints, and an objective.
class LP {
public:
    vector<Variable> variables;
    vector<Constraint> constraints;
    Objective objective;

    LP(bool maximize) : objective(maximize) {}

    // Add a variable to the LP
    int add_variable(const string& name, double lower_bound, double upper_bound) {
        variables.push_back({name, lower_bound, upper_bound});
        return variables.size() - 1;
    }

    // Add a constraint to the LP
    int add_constraint(const string& name, double lower_bound, double upper_bound) {
        constraints.push_back({name, lower_bound, upper_bound});
        return constraints.size() - 1;
    }

    // Add `coefficient * variable` to the given constraint
    void add_to_constraint(int c, int v, double a) {
        constraints[c].sum[v] += a;
    }

    // Add `coefficient * variable` to the objective
    void add_to_objective(int v, double a) {
        objective.sum[v] += a;
    }
};

ostream& operator<<(ostream& os, const Variable& v) {
    if (v.lower_bound != -INFINITY)
        os << v.lower_bound << " <= ";
    os << v.name;
    if (v.upper_bound != INFINITY)
        os << " <= " << v.upper_bound;
    return os;
}

ostream& print_sum(
    ostream& os, const unordered_map<int, double>& sum, const vector<Variable>& variables
) {
    map<const string, double> sorted_sum;
    for (const auto& t : sum) {
        sorted_sum[variables[t.first].name] = t.second;
    }
    bool first = true;
    for (const auto& t : sorted_sum) {
        if (t.second == 0)
            continue;
        if (!first)
            os << " + ";
        first = false;
        os << t.second << " " << t.first;
    }
    return os;
}

ostream& print_constraint(
    ostream& os, const Constraint& c, const vector<Variable>& variables
) {
    os << c.name << ": ";
    if (c.lower_bound != -INFINITY)
        os << c.lower_bound << " <= ";
    print_sum(os, c.sum, variables);
    if (c.upper_bound != INFINITY)
        os << " <= " << c.upper_bound;
    return os;
}

ostream& operator<<(ostream& os, const LP& lp) {
    os << (lp.objective.maximize ? "maximize" : "minimize") << " ";
    print_sum(os, lp.objective.sum, lp.variables) << endl;
    os << "subject to" << endl;
    for (const auto& v : lp.variables)
        os << "    " << v << endl;
    map<const string, const Constraint> sorted_constraints;
    for (const auto& c : lp.constraints) {
        sorted_constraints.insert({c.name, c});
    }
    for (const auto& t : sorted_constraints) {
        os << "    ";
        print_constraint(os, t.second, lp.variables) << endl;
    }
    return os;
}

int main() {
    unordered_map<int, double> m;
    m[1] += 1;
    cout << m[1] << endl;
    return 0;
}
