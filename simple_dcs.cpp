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
        os << t.second << "*" << t.first;
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

// Convert an LP to a HiGHs LpProblem
pair<double,vector<double>> solve(const LP &p) {
    int n = p.variables.size();
    int m = p.constraints.size();

    HighsModel model;
    model.lp_.num_col_ = n;
    model.lp_.num_row_ = m;
    model.lp_.sense_ = p.objective.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    model.lp_.offset_ = 0.0;
    model.lp_.col_cost_.resize(n);
    for (const auto& v : p.objective.sum)
        model.lp_.col_cost_[v.first] = v.second;
    model.lp_.col_lower_.resize(n);
    model.lp_.col_upper_.resize(n);
    int i = 0;
    for (const auto& v : p.variables) {
        model.lp_.col_lower_[i] = v.lower_bound;
        model.lp_.col_upper_[i] = v.upper_bound;
        ++i;
    }
    model.lp_.row_lower_.resize(m);
    model.lp_.row_upper_.resize(m);
    i = 0;
    for (const auto& c : p.constraints) {
        model.lp_.row_lower_[i] = c.lower_bound;
        model.lp_.row_upper_[i] = c.upper_bound;
        ++i;
    }
    model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
    model.lp_.a_matrix_.start_.resize(m + 1);
    i = 0;
    for (const auto& c : p.constraints) {
        model.lp_.a_matrix_.start_[i] = model.lp_.a_matrix_.index_.size();
        for (const auto& t : c.sum) {
            model.lp_.a_matrix_.index_.push_back(t.first);
            model.lp_.a_matrix_.value_.push_back(t.second);
        }
        ++i;
    }
    model.lp_.a_matrix_.start_[i] = model.lp_.a_matrix_.index_.size();

    // Create a Highs instance
    Highs highs;
    HighsStatus return_status;

    // Pass the model to HiGHS
    return_status = highs.passModel(model);
    assert(return_status==HighsStatus::kOk);

    // Get a const reference to the LP data in HiGHS
    // const HighsLp& lp = highs.getLp();

    // Solve the model
    return_status = highs.run();
    assert(return_status==HighsStatus::kOk);

    // Get the model status
    const HighsModelStatus& model_status = highs.getModelStatus();
    assert(model_status==HighsModelStatus::kOptimal);

    const HighsInfo& info = highs.getInfo();
    const double obj = info.objective_function_value;
    // const bool has_values = info.primal_solution_status;
    const HighsSolution& solution = highs.getSolution();

    return make_pair(obj, solution.col_value);
}

void test_lp1() {
    LP lp(true);
    int x = lp.add_variable("x", 0.0, INFINITY);
    int y = lp.add_variable("y", 0.0, INFINITY);
    int z = lp.add_variable("z", 0.0, INFINITY);
    int t = lp.add_variable("t", 0.0, 2.0);
    int c1 = lp.add_constraint("c1", -INFINITY, 1.0);
    lp.add_to_constraint(c1, x, 1.0);
    lp.add_to_constraint(c1, y, 1.0);
    int c2 = lp.add_constraint("c2", -INFINITY, 1.0);
    lp.add_to_constraint(c2, y, 1.0);
    lp.add_to_constraint(c2, z, 1.0);
    int c3 = lp.add_constraint("c3", -INFINITY, 1.0);
    lp.add_to_constraint(c3, x, 1.0);
    lp.add_to_constraint(c3, z, 1.0);
    lp.add_to_objective(x, 1.0);
    lp.add_to_objective(y, 1.0);
    lp.add_to_objective(z, 1.0);
    lp.add_to_objective(t, 1.0);
    cout << lp << endl;
    auto sol = solve(lp);
    auto& obj = sol.first;
    auto& val = sol.second;
    assert(abs(obj - 3.5) < 1e-7);
    assert(abs(val[x] - 0.5) < 1e-7);
    assert(abs(val[y] - 0.5) < 1e-7);
    assert(abs(val[z] - 0.5) < 1e-7);
    assert(abs(val[t] - 2.0) < 1e-7);
}

int main() {
    test_lp1();
    return 0;
}
