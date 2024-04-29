#include <iostream>
#include <vector>
#include <set>
#include <map>
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
    map<string, double> sum;
};

// The objective has a boolean flag to indicate whether it is a maximization or minimization
// and a sum indicating the objective function. The sum is represented as a dictionary from
// variables names to coefficients.
struct Objective {
    bool maximize;
    map<string, double> sum;

    Objective(bool maximize_) : maximize(maximize_) {}
};

// A linear program has variables, constraints, and an objective.
class LP {
public:
    map<string, Variable> variables;
    map<string, Constraint> constraints;
    Objective objective;

    LP(bool maximize) : objective(maximize) {}

    // Add a variable to the LP
    void add_variable(const string& name, double lower_bound, double upper_bound) {
        assert(variables.find(name) == variables.end());
        variables.insert({name, {name, lower_bound, upper_bound}});
    }

    // Add a constraint to the LP
    void add_constraint(const string& name, double lower_bound, double upper_bound) {
        assert(constraints.find(name) == constraints.end());
        constraints.insert({name, {name, lower_bound, upper_bound}});
    }

    // Add `coefficient * variable` to the given constraint
    void add_to_constraint(const string& constraint, const string& variable, double coefficient) {
        assert(constraints.find(constraint) != constraints.end());
        assert(variables.find(variable) != variables.end());
        constraints[constraint].sum[variable] += coefficient;
    }

    // Add `coefficient * variable` to the objective
    void add_to_objective(const string& variable, double coefficient) {
        assert(variables.find(variable) != variables.end());
        objective.sum[variable] += coefficient;
    }
};

std::ostream& operator<<(std::ostream& os, const Variable& v) {
    if (v.lower_bound != -INFINITY)
        os << v.lower_bound << " <= ";
    os << v.name;
    if (v.upper_bound != INFINITY)
        os << " <= " << v.upper_bound;
    return os;
}

std::ostream& operator<<(std::ostream& os, const map<string, double>& sum) {
    bool first = true;
    for (const auto& t : sum) {
        if (!first)
            os << " + ";
        first = false;
        os << t.second << "*" << t.first;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Constraint& constraint) {
    os << constraint.name << ":    ";
    if (constraint.lower_bound != -INFINITY)
        os << constraint.lower_bound << " <= ";
    os << constraint.sum;
    if (constraint.upper_bound != INFINITY)
        os << " <= " << constraint.upper_bound;
    return os;
}

std::ostream& operator<<(std::ostream& os, const Objective& obj) {
    os << (obj.maximize ? "maximize" : "minimize") << " ";
    os << obj.sum;
    return os;
}

std::ostream& operator<<(std::ostream& os, const LP& lp) {
    os << lp.objective << endl;
    os << "subject to" << endl;
    for (const auto& v : lp.variables) {
        os << "    " << v.second << endl;
    }
    for (const auto& c : lp.constraints) {
        os << "    " << c.second << endl;
    }
    return os;
}

inline float g(float x) {
    if (x == INFINITY)
        return 1.0e30;
    if (x == -INFINITY)
        return -1.0e30;
    return x;
}

// Convert an LP to a Pulp LpProblem
pair<double,map<string,double>> solve(LP &p) {
    map<string, double> var_map;
    int n = p.variables.size();
    int m = p.constraints.size();
    map<string, int> var_index;
    int i = 0;
    for (const auto& v : p.variables)
        var_index[v.first] = i++;
    HighsModel model;
    model.lp_.num_col_ = n;
    model.lp_.num_row_ = m;
    model.lp_.sense_ = p.objective.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    model.lp_.offset_ = 0.0;
    model.lp_.col_cost_.resize(n);
    for (const auto& v : p.objective.sum)
        model.lp_.col_cost_[var_index[v.first]] = v.second;
    model.lp_.col_lower_.resize(n);
    model.lp_.col_upper_.resize(n);
    i = 0;
    for (const auto& v : p.variables) {
        model.lp_.col_lower_[i] = g(v.second.lower_bound);
        model.lp_.col_upper_[i] = g(v.second.upper_bound);
        ++i;
    }
    model.lp_.row_lower_.resize(m);
    model.lp_.row_upper_.resize(m);
    i = 0;
    for (const auto& c : p.constraints) {
        model.lp_.row_lower_[i] = g(c.second.lower_bound);
        model.lp_.row_upper_[i] = g(c.second.upper_bound);
        ++i;
    }
    model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
    model.lp_.a_matrix_.start_.resize(m + 1);
    i = 0;
    for (const auto& c : p.constraints) {
        model.lp_.a_matrix_.start_[i] = model.lp_.a_matrix_.index_.size();
        for (const auto& t : c.second.sum) {
            model.lp_.a_matrix_.index_.push_back(var_index[t.first]);
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
    const float obj = info.objective_function_value;
    const bool has_values = info.primal_solution_status;

    const HighsSolution& solution = highs.getSolution();
    map<string,double> sol;
    // cout << "Primal Solution:" << endl;
    i = 0;
    if (has_values)
        for (auto& v : p.variables) {
            sol[v.first] = solution.col_value[i++];
            // cout << "    " << v.first << ": " << sol[v.first] << endl;
        }

    return make_pair(obj, sol);
}

void test_lp1() {
    LP lp(true);
    lp.add_variable("x", 0.0, INFINITY);
    lp.add_variable("y", 0.0, INFINITY);
    lp.add_variable("z", 0.0, INFINITY);
    lp.add_variable("t", 0.0, 2.0);
    lp.add_constraint("c1", -INFINITY, 1.0);
    lp.add_to_constraint("c1", "x", 1.0);
    lp.add_to_constraint("c1", "y", 1.0);
    lp.add_constraint("c2", -INFINITY, 1.0);
    lp.add_to_constraint("c2", "y", 1.0);
    lp.add_to_constraint("c2", "z", 1.0);
    lp.add_constraint("c3", -INFINITY, 1.0);
    lp.add_to_constraint("c3", "x", 1.0);
    lp.add_to_constraint("c3", "z", 1.0);
    lp.add_to_objective("x", 1.0);
    lp.add_to_objective("y", 1.0);
    lp.add_to_objective("z", 1.0);
    lp.add_to_objective("t", 1.0);
    float opt = solve(lp).first;
    assert(abs(opt - 3.5) < 1e-7);
}

/******************************************************************************************/

// A representation of a bound on an Lp-norm of a degree sequence
struct DC {
    set<string> X;
    set<string> Y;
    double p;
    double b;

    DC(const set<string>& X_, const set<string>& Y_, double p_, double b_)
    : X(X_), Y(Y_), p(p_), b(b_) {
        copy(X.begin(), X.end(), inserter(Y, Y.end()));
    }
};

// Given a set `X`, convert each element to a string and concatenate the strings.
inline string _name(const set<string> &X) {
    string name = "{";
    bool first = true;
    for (const auto &x : X) {
        if (!first)
            name += ",";
        first = false;
        name += x;
    }
    name += "}";
    return name;
}

ostream& operator<<(ostream& os, const DC& dc) {
    os << "log_2 |deg(" << _name(dc.Y) << "|" << _name(dc.X) << ")|_" << dc.p << " <= " << dc.b;
    return os;
}

inline string flow_var_name(int t, const set<string> &X, const set<string> &Y) {
    return "f" + to_string(t) + "_" + _name(X) + "->" + _name(Y);
}

inline string flow_capacity_name(int t, const set<string> &X, const set<string> &Y) {
    return "c" + to_string(t) + "_" + _name(X) + "->" + _name(Y);
}

inline string flow_conservation_name(int t, const set<string> &Z) {
    return "e" + to_string(t) + "_" + _name(Z);
}

inline string dc_coefficient_name(int i) {
    return "a_" + to_string(i);
}

bool is_subset(const set<string> &X, const set<string> &Y) {
    return includes(Y.begin(), Y.end(), X.begin(), X.end());
}

void _add_edge(
    const set<string>& X,
    const set<string>& Y,
    set<set<string>>& vertices,
    set<pair<set<string>, set<string>>>& edges
) {
    assert(is_subset(X, Y) || is_subset(Y, X));
    if (X != Y && edges.find({Y, X}) == edges.end()) {
        vertices.insert(X);
        vertices.insert(Y);
        edges.insert({X, Y});
    }
}

// Given a list of DCs `dcs` and a list of target variables `vars`, construct the vertices
// and edges of the network flow.
pair<set<set<string>>, set<pair<set<string>, set<string>>>>
_collect_vertices_and_edges(const vector<DC> &dcs, const vector<string> &vars) {
    set<set<string>> vertices;
    set<pair<set<string>, set<string>>> edges;

    // For every DC, add edges to the network flow
    for (auto &dc : dcs) {
        // Add edge from X to Y
        _add_edge(dc.X, dc.Y, vertices, edges);

        // If p is not infinity, add edge from {} to X
        if (dc.p != INFINITY)
            _add_edge({}, dc.X, vertices, edges);
    }

    // Add edges from Y to {} and from Y to {y} for every y in Y
    for (auto &dc : dcs) {
        _add_edge(dc.Y, {}, vertices, edges);
        for (auto &y : dc.Y) {
            _add_edge(dc.Y, {y}, vertices, edges);
        }
    }

    return make_pair(vertices, edges);
}

void add_flow_constraints(
    LP &lp,
    const vector<DC>& dcs,
    const vector<string>& vars,
    const set<set<string>>& vertices,
    const set<pair<set<string>, set<string>>>& edges
) {
    for (size_t i = 0; i < dcs.size(); ++i) {
        assert(dcs[i].X.size() <= 1); // Only simple degree constraints are supported
        lp.add_variable(dc_coefficient_name(i), 0.0, INFINITY);
    }

    // For every t-flow
    for (size_t t = 0; t < vars.size(); ++t) {
        // For every vertex `Z`, add a flow conservation constraint
        for (const auto &Z : vertices) {
            double nt_Z = (Z.size() == 1 && *Z.begin() == vars[t]) ?
                1.0 : (Z.size() == 0 ? -1.0 : 0.0);
            lp.add_constraint(flow_conservation_name(t, Z), nt_Z, INFINITY);
        }

        // For every edge `X -> Y`
        for (const auto &edge : edges) {
            auto &X = edge.first;
            auto &Y = edge.second;
            const string& ft_X_Y = flow_var_name(t, X, Y);
            double lower_bound = (X.size() <= Y.size()) ? -INFINITY : 0.0;
            double upper_bound = INFINITY;
            lp.add_variable(ft_X_Y, lower_bound, upper_bound);
            lp.add_to_constraint(flow_conservation_name(t, X), ft_X_Y, -1.0);
            lp.add_to_constraint(flow_conservation_name(t, Y), ft_X_Y, 1.0);
            if (X.size() <= Y.size()) {
                const string& ct_X_Y = flow_capacity_name(t, X, Y);
                lp.add_constraint(ct_X_Y, 0.0, INFINITY);
                lp.add_to_constraint(ct_X_Y, ft_X_Y, -1.0);
            }
        }

        // Add coefficients to capacity constraints
        for (size_t i = 0; i < dcs.size(); ++i) {
            string a_i = dc_coefficient_name(i);
            if (dcs[i].X.size() != dcs[i].Y.size()) {
                lp.add_to_constraint(flow_capacity_name(t, dcs[i].X, dcs[i].Y), a_i, 1.0);
            }
            if (dcs[i].p != INFINITY && !dcs[i].X.empty()) {
                lp.add_to_constraint(flow_capacity_name(t, {}, dcs[i].X), a_i, 1.0 / dcs[i].p);
            }
        }
    }
}

void set_objective(LP &lp, const vector<DC> &dcs) {
    for (size_t i = 0; i < dcs.size(); ++i) {
        lp.add_to_objective(dc_coefficient_name(i), dcs[i].b);
    }
}

double simple_dc_bound(vector<DC> &dcs, const vector<string> &vars) {
    LP lp(false);
    auto ve = _collect_vertices_and_edges(dcs, vars);
    auto &vertices = ve.first;
    auto &edges = ve.second;
    add_flow_constraints(lp, dcs, vars, vertices, edges);
    set_objective(lp, dcs);
    return solve(lp).first;
}

// Testcases for the simple_dc_bound function
void test_simple_dc_bound1() {
    vector<DC> dcs = {
        { {}, {"A", "B"}, 1, 1 },
        { {}, {"A", "C"}, 1, 1 },
        { {}, {"B", "C"}, 1, 1 }
    };
    vector<string> vars = { "A", "B", "C" };
    float p = simple_dc_bound(dcs, vars);
    assert(abs(p - 1.5) < 1e-7);
}

void test_simple_dc_bound2() {
    vector<DC> dcs = {
        { {}, {"A", "B"}, INFINITY, 1 },
        { {}, {"A", "C"}, INFINITY, 1 },
        { {}, {"B", "C"}, INFINITY, 1 }
    };
    vector<string> vars = { "A", "B", "C" };
    float p = simple_dc_bound(dcs, vars);
    assert(abs(p - 1.5) < 1e-7);
}

void test_simple_dc_bound3() {
    vector<DC> dcs = {
        { {"A"}, {"B"}, 2, 1 },
        { {"B"}, {"C"}, 2, 1 },
        { {"C"}, {"A"}, 2, 1 }
    };
    vector<string> vars = { "A", "B", "C" };
    float p = simple_dc_bound(dcs, vars);
    assert(abs(p - 2.0) < 1e-7);
}

// Add other test functions similarly
int main() {
    test_lp1();
    cout << string(80, '-') << endl;
    test_simple_dc_bound1();
    cout << string(80, '-') << endl;
    test_simple_dc_bound2();
    cout << string(80, '-') << endl;
    test_simple_dc_bound3();
    cout << string(80, '-') << endl;
    return 0;
}
