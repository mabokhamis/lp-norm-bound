#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <cmath>
#include <numeric>
#include <tuple>
#include <cassert>
#include <chrono>
#include "Highs.h"

using namespace std;


/******************************************************************************************/
// NOTE: The main entry point to this file is the `flow_bound` function below
/******************************************************************************************/


/******************************************************************************************/
// The following macro should be left undefined unless you want to enable debug mode:
// #define DEBUG
/******************************************************************************************/


//==========================================================================================
// Generic interface for building and solving LPs using HiGHS
//==========================================================================================

// A variable has a name, lower bound, and upper bound.
struct Variable {
    const string name;
    const double lower_bound;
    const double upper_bound;
};

// A constraint has a name, lower bound, upper bound, and a sum of variables. The sum is
// represented as a dictionary from variable names to coefficients.
struct Constraint {
    const string name;
    const double lower_bound;
    const double upper_bound;
    unordered_map<int, double> sum;
};

// The objective has a boolean flag to indicate whether it is a maximization or minimization
// and a sum indicating the objective function. The sum is represented as a dictionary from
// variables names to coefficients.
struct Objective {
    const bool maximize;
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
        constraints.at(c).sum[v] += a;
    }

    // Add `coefficient * variable` to the objective
    void add_to_objective(int v, double a) {
        objective.sum[v] += a;
    }
};

#ifdef DEBUG
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
    os << "Linear Program:" << endl;
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
#endif

// Convert an LP to a HiGHs LpProblem
pair<double,vector<double>> solve(const LP &p) {
    #ifdef DEBUG
    cout << p << endl;
    #endif
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
    #ifndef DEBUG
    // Set the options to reduce verbosity
    highs.setOptionValue("output_flag", false);  // Disables all output
    highs.setOptionValue("log_to_console", false);  // Specifically disables logging to the console
    highs.setOptionValue("message_level", 0);  // Disables all messages
    #endif
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
    const bool has_values = info.primal_solution_status;
    assert(has_values);
    const HighsSolution& solution = highs.getSolution();

    return make_pair(obj, solution.col_value);
}

// A testcase for the LP interface
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
    auto sol = solve(lp);
    auto& obj = sol.first;
    auto& val = sol.second;
    assert(abs(obj - 3.5) < 1e-7);
    assert(abs(val[x] - 0.5) < 1e-7);
    assert(abs(val[y] - 0.5) < 1e-7);
    assert(abs(val[z] - 0.5) < 1e-7);
    assert(abs(val[t] - 2.0) < 1e-7);
}



//==========================================================================================
// Generic utility functions:
//==========================================================================================

template <typename T>
set<T> set_union(const set<T>& X, const set<T>& Y) {
    set<T> Z;
    copy(X.begin(), X.end(), inserter(Z, Z.end()));
    copy(Y.begin(), Y.end(), inserter(Z, Z.end()));
    return Z;
}

template <typename T>
set<T> set_difference(const set<T>& X, const set<T>& Y) {
    set<T> Z;
    set_difference(X.begin(), X.end(), Y.begin(), Y.end(), inserter(Z, Z.end()));
    return Z;
}

template <typename T>
bool is_subset(const set<T>& X, const set<T>& Y) {
    return includes(Y.begin(), Y.end(), X.begin(), X.end());
}



//==========================================================================================
// The flow bound implementation with Lp-norm constraints
//==========================================================================================

// A representation of a bound on an Lp-norm of a degree sequence
template <typename T>
struct DC {
    const set<T> X;
    const set<T> Y;
    const double p;
    const double b;

    DC(const set<T>& X_, const set<T>& Y_, double p_, double b_)
        : X(X_), Y(set_union(X_, Y_)), p(p_), b(b_) {}
};

// In Debug mode, we generate meaningful names for variables and constraints.
#ifdef DEBUG
inline string name(string s) {
    return s;
}

inline string name(int i) {
    return to_string(i);
}

// Given a set `X`, convert each element to a string and concatenate the strings.
template <typename T>
inline string set_name(const set<T> &X) {
    string s = "{";
    bool first = true;
    for (const auto &x : X) {
        if (!first)
            s += ",";
        first = false;
        s += name(x);
    }
    s += "}";
    return s;
}

template <typename T>
ostream& operator<<(ostream& os, const DC<T>& dc) {
    os << "log_2 ||deg(" << set_name(dc.Y) << "|" << set_name(dc.X) << ")||_" <<
        dc.p << " <= " << dc.b;
    return os;
}

template <typename T>
inline string flow_var_name(int t, const set<T> &X, const set<T> &Y) {
    return "f" + to_string(t) + "_" + set_name(X) + "->" + set_name(Y);
}

template <typename T>
inline string flow_capacity_con_name(int t, const set<T> &X, const set<T> &Y) {
    return "c" + to_string(t) + "_" + set_name(X) + "->" + set_name(Y);
}

template <typename T>
inline string flow_conservation_con_name(int t, const set<T> &Z) {
    return "e" + to_string(t) + "_" + set_name(Z);
}

inline string dc_var_name(int i) {
    return "a" + to_string(i);
}
// However, in Release mode, we skip name generation and just use empty strings
#else
template <typename T>
inline string flow_var_name(int t, const set<T> &X, const set<T> &Y) {
    return "";
}

template <typename T>
inline string flow_capacity_con_name(int t, const set<T> &X, const set<T> &Y) {
    return "";
}

template <typename T>
inline string flow_conservation_con_name(int t, const set<T> &Z) {
    return "";
}

inline string dc_var_name(int i) {
    return "";
}
#endif

// Given a directed multi-graph `G` (i.e. where we can have multiple edges `u -> v`):
//  - if `G` is acyclic, return `true` along with a topological ordering of the vertices
//  - otherwise, return `false` along an ordering of the vertices that is as close to a
//    topological ordering as possible
template <typename T>
pair<bool,vector<T>> approximate_topological_sort(
    const vector<T>& V, const vector<pair<T, T>>& E
) {
    map<T, int> out_degree;
    map<T, vector<T>> in_edges;
    for (auto& v : V)
        out_degree[v] = 0;
    for (auto& e : E) {
        // Every edge must have both endpoints in V
        assert(out_degree.find(e.first) != out_degree.end());
        assert(out_degree.find(e.second) != out_degree.end());

        out_degree[e.first]++;
        in_edges[e.second].push_back(e.first);
    }
    set<pair<int, T>> Q;
    for (auto& p : out_degree) {
        Q.insert({p.second, p.first});
    }
    bool acyclic = true;
    vector<T> order;
    while (!Q.empty()) {
        auto it = Q.begin();
        if (it->first != 0)
            acyclic = false;
        T u = it->second;
        order.push_back(u);
        out_degree.erase(u);
        Q.erase(it);
        for (auto& v : in_edges[u])
            if (out_degree.find(v) != out_degree.end()) {
                assert(out_degree[v] > 0);
                out_degree[v]--;
                Q.erase({out_degree[v] + 1, v});
                Q.insert({out_degree[v], v});
            }
    }
    reverse(order.begin(), order.end());
    return {acyclic, order};
}

// Instead of a directed multi-graph, now we are given a set of DCs
template <typename T>
pair<bool,vector<T>> approximate_topological_sort(
    const vector<T>& V, const vector<DC<T>>& dcs
) {
    vector<pair<T, T>> E;
    for (auto& dc : dcs)
        for (auto& x : dc.X)
            for (auto& y : set_difference(dc.Y, dc.X))
                E.push_back({x, y});
    return approximate_topological_sort(V, E);
}

template <typename T>
struct LpNormLP {
    const vector<DC<T>> dcs;
    const vector<T> vars;

    bool use_only_chain_bound;
    vector<DC<T>> simple_dcs;
    vector<DC<T>> non_simple_dcs;
    bool is_acyclic;
    vector<T> var_order;
    map<T, int> var_index;

    LP lp;
    map<tuple<int, const set<T>, const set<T>>, int> flow_var;
    map<tuple<int, const set<T>, const set<T>>, int> flow_capacity_con;
    map<tuple<int, const set<T>>, int> flow_conservation_con;
    map<int, int> dc_var;

    set<set<T>> vertices;
    set<pair<set<T>, set<T>>> edges;

    LpNormLP(
        const vector<DC<T>>& dcs_, const vector<T>& vars_, bool use_only_chain_bound_
    ) : dcs(dcs_), vars(vars_), use_only_chain_bound(use_only_chain_bound_), lp(false) {

        verify_input();

        // If all degrees are acyclic (including the simple ones), then it seems more efficient
        // to just use the chain bound
        if (approximate_topological_sort(vars, dcs).first)
            use_only_chain_bound = true;

        for (const auto& dc : dcs)
            if (is_simple(dc))
                simple_dcs.push_back(dc);
            else
                non_simple_dcs.push_back(dc);
        auto p = approximate_topological_sort(vars, non_simple_dcs);
        is_acyclic = p.first;
        var_order = p.second;
        for (size_t i = 0; i < var_order.size(); ++i) {
            var_index[var_order[i]] = i;
        }
    }

    // Verify the validity of the input DCs and vars
    void verify_input() {
        set<T> var_set;
        for (const auto& v : vars) {
            // If the following assertion fails, this means that the input vars are not unique
            assert(var_set.find(v) == var_set.end());

            var_set.insert(v);
        }
        for (const auto& dc : dcs) {
            assert(dc.p > 0.0);
            assert(dc.b >= 0.0);
            assert(is_subset(dc.X, var_set));
            assert(is_subset(dc.Y, var_set));
        }
    }

    bool is_simple(const DC<T>& dc) {
        return !use_only_chain_bound && dc.X.size() <= 1;
    }

    DC<T> order_consistent_dc(const DC<T>& dc) {
        int last_x = -1;
        for (const auto& x : dc.X)
            last_x = max(last_x, var_index.at(x));
        set<T> new_Y;
        for (const auto& y : dc.Y)
            if (var_index.at(y) > last_x)
                new_Y.insert(y);
        return DC<T>(dc.X, new_Y, dc.p, dc.b);
    }

    int add_flow_var(
        int t, const set<T>& X, const set<T>& Y, double lower_bound, double upper_bound
    ) {
        auto key = make_tuple(t, X, Y);
        auto it = flow_var.find(key);
        assert(it == flow_var.end());
        int v = lp.add_variable(flow_var_name(t, X, Y), lower_bound, upper_bound);
        flow_var.insert({key, v});
        return v;
    }

    inline int get_flow_var(int t, const set<T>& X, const set<T>& Y) {
        auto key = make_tuple(t, X, Y);
        return flow_var.at(key);
    }

    int add_flow_capacity_con(
        int t, const set<T>& X, const set<T>& Y, double lower_bound, double upper_bound
    ) {
        auto key = make_tuple(t, X, Y);
        auto it = flow_capacity_con.find(key);
        assert(it == flow_capacity_con.end());
        int c = lp.add_constraint(flow_capacity_con_name(t, X, Y), lower_bound, upper_bound);
        flow_capacity_con.insert({key, c});
        return c;
    }

    inline int get_flow_capacity_con(int t, const set<T>& X, const set<T>& Y) {
        auto key = make_tuple(t, X, Y);
        return flow_capacity_con.at(key);
    }

    int add_flow_conservation_con(
        int t, const set<T>& Z, double lower_bound, double upper_bound
    ) {
        auto key = make_tuple(t, Z);
        auto it = flow_conservation_con.find(key);
        assert(it == flow_conservation_con.end());
        int c = lp.add_constraint(flow_conservation_con_name(t, Z), lower_bound, upper_bound);
        flow_conservation_con.insert({key, c});
        return c;
    }

    inline int get_flow_conservation_con(int t, const set<T>& Z) {
        auto key = make_tuple(t, Z);
        return flow_conservation_con.at(key);
    }

    int add_dc_var(int i, double lower_bound, double upper_bound) {
        auto it = dc_var.find(i);
        assert(it == dc_var.end());
        int v = lp.add_variable(dc_var_name(i), lower_bound, upper_bound);
        dc_var.insert({i, v});
        return v;
    }

    inline int get_dc_var(int i) {
        return dc_var.at(i);
    }

    void add_edge(const set<T>& X, const set<T>& Y) {
        assert(is_subset(X, Y) || is_subset(Y, X));
        if (X != Y && edges.find({Y, X}) == edges.end()) {
            vertices.insert(X);
            vertices.insert(Y);
            edges.insert({X, Y});
        }
    }

    void construct_graph() {
        for (auto v : vars)
            vertices.insert({v});
        for (auto &dc : simple_dcs) {
            // Add edge from X to Y
            add_edge(dc.X, dc.Y);

            // If p is not infinity, add an edge from {} to X
            if (dc.p != INFINITY)
                add_edge({}, dc.X);
        }

        // Add edges from Y to {} and from Y to {y} for every y in Y
        for (auto &dc : simple_dcs) {
            add_edge(dc.Y, {});
            for (auto &y : dc.Y) {
                add_edge(dc.Y, {y});
            }
        }
    }

    void add_flow_constraints() {
        for (size_t i = 0; i < dcs.size(); ++i) {
            add_dc_var(i, 0.0, INFINITY);
        }

        // For every t-flow
        for (size_t t = 0; t < vars.size(); ++t) {
            // For every vertex `Z`, add a flow conservation constraint
            for (const auto &Z : vertices) {
                if (Z.empty())
                    continue;
                double lower_bound = (Z.size() == 1 && *Z.begin() == vars[t]) ? 1.0 : 0.0;
                add_flow_conservation_con(t, Z, lower_bound, INFINITY);
            }

            // For every edge `X -> Y`
            for (const auto &edge : edges) {
                auto &X = edge.first;
                auto &Y = edge.second;
                double lower_bound = (X.size() <= Y.size()) ? -INFINITY : 0.0;
                double upper_bound = INFINITY;
                int ft_X_Y = add_flow_var(t, X, Y, lower_bound, upper_bound);
                if (!X.empty())
                    lp.add_to_constraint(get_flow_conservation_con(t, X), ft_X_Y, -1.0);
                if (!Y.empty())
                    lp.add_to_constraint(get_flow_conservation_con(t, Y), ft_X_Y, 1.0);
                if (X.size() <= Y.size()) {
                    int ct_X_Y = add_flow_capacity_con(t, X, Y, 0.0, INFINITY);
                    lp.add_to_constraint(ct_X_Y, ft_X_Y, -1.0);
                }
            }

            for (size_t i = 0; i < dcs.size(); ++i) {
                int ai = get_dc_var(i);
                // For simple DCs, add coefficients to capacity constraints
                if (is_simple(dcs[i])) {
                    if (dcs[i].X.size() != dcs[i].Y.size()) {
                        lp.add_to_constraint(
                            get_flow_capacity_con(t, dcs[i].X, dcs[i].Y), ai, 1.0);
                    }
                    if (dcs[i].p != INFINITY && !dcs[i].X.empty()) {
                        lp.add_to_constraint(
                            get_flow_capacity_con(t, {}, dcs[i].X), ai, 1.0 / dcs[i].p);
                    }
                }
                // For acyclic DCs, add coefficients to flow conservation constraints
                else {
                    DC<T> dc = order_consistent_dc(dcs[i]);
                    for (const auto& y : set_difference(dc.Y, dc.X)) {
                        int e_y = get_flow_conservation_con(t, {y});
                        lp.add_to_constraint(e_y, ai, 1.0);
                    }
                    if (dc.p != INFINITY)
                        for (const auto& x : dc.X) {
                            int e_x = get_flow_conservation_con(t, {x});
                            lp.add_to_constraint(e_x, ai, 1.0 / dc.p);
                        }
                }
            }
        }
    }

    void set_objective() {
        for (size_t i = 0; i < dcs.size(); ++i) {
            lp.add_to_objective(get_dc_var(i), dcs[i].b);
        }
    }
};

// In Release mode, we convert the strings used to represent variables to integers
#ifndef DEBUG
// Convert the type of the given DC vars from T1 to T2, and return the resulting DC
template <typename T1, typename T2>
DC<T2> transform_DC(const DC<T1>& dc, const map<T1, T2>& f) {
    set<T2> new_X, new_Y;

    for (const auto& x : dc.X) {
        new_X.insert(f.at(x));
    }
    for (const auto& y : dc.Y) {
        new_Y.insert(f.at(y));
    }
    return DC<T2>(new_X, new_Y, dc.p, dc.b);
}

// Convert DCs from string to integer representation
pair<vector<DC<int>>, vector<int>> transform_dcs_to_int(
    const vector<DC<string>>& dcs, const vector<string>& vars
) {
    map<string, int> var_map;
    for (size_t i = 0; i < vars.size(); ++i) {
        var_map[vars[i]] = i;
    }

    vector<DC<int>> new_dcs;
    vector<int> new_vars;
    for (const auto& dc : dcs) {
        new_dcs.push_back(transform_DC(dc, var_map));
    }

    for (const auto& v : vars) {
        new_vars.push_back(var_map[v]);
    }

    return make_pair(new_dcs, new_vars);
}
#endif

/*************************************************/
// NOTE: This is the main entry point to this file
/*************************************************/
double flow_bound(
    const vector<DC<string>> &dcs,
    const vector<string> &vars,

    // This optional parameter is for testing purposes only. If set to true, it will restrict
    // the flow bound to become the weaker chain bound.
    bool use_only_chain_bound = false
) {
    // In debug mode, we operate directly on the given strings used to represent variables
    #ifdef DEBUG
    cout << "Degree Constraints:" << endl;
    for (const auto& dc : dcs) {
        cout << "    " << dc << endl;
    }
    cout << endl;
    LpNormLP<string> lp(dcs, vars, use_only_chain_bound);
    // In release mode, we convert the strings to integers
    #else
    auto int_dcs_vars = transform_dcs_to_int(dcs, vars);
    LpNormLP<int> lp(int_dcs_vars.first, int_dcs_vars.second, use_only_chain_bound);
    #endif
    lp.construct_graph();
    lp.add_flow_constraints();
    lp.set_objective();
    return solve(lp.lp).first;
}



//==========================================================================================
// Testcases for the flow_bound function
//==========================================================================================

void test_flow_bound1() {
    vector<DC<string>> dcs = {
        { {}, {"A", "B"}, 1, 1 },
        { {}, {"A", "C"}, 1, 1 },
        { {}, {"B", "C"}, 1, 1 }
    };
    vector<string> vars = { "A", "B", "C" };
    double p;
    p = flow_bound(dcs, vars, false);
    assert(abs(p - 1.5) < 1e-7);
    p = flow_bound(dcs, vars, true);
    assert(abs(p - 1.5) < 1e-7);
}

void test_flow_bound2() {
    vector<DC<string>> dcs = {
        { {}, {"A", "B"}, INFINITY, 1 },
        { {}, {"A", "C"}, INFINITY, 1 },
        { {}, {"B", "C"}, INFINITY, 1 }
    };
    vector<string> vars = { "A", "B", "C" };
    double p;
    p = flow_bound(dcs, vars, false);
    assert(abs(p - 1.5) < 1e-7);
    p = flow_bound(dcs, vars, true);
    assert(abs(p - 1.5) < 1e-7);
}

void test_flow_bound3() {
    vector<DC<string>> dcs = {
        { {"A"}, {"B"}, 2, 1 },
        { {"B"}, {"C"}, 2, 1 },
        { {"C"}, {"A"}, 2, 1 }
    };
    vector<string> vars = { "A", "B", "C" };
    double p;
    p = flow_bound(dcs, vars, false);
    assert(abs(p - 2.0) < 1e-7);
    p = flow_bound(dcs, vars, true);
    assert(abs(p - 3.0) < 1e-7);
}

void test_flow_bound4() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 1 },
        { {}, {"y", "z"}, 1, 1 },
        { {}, {"x", "z"}, 1, 1 },
        { {"x", "z"}, {"u"}, INFINITY, 0},
        { {"y", "z"}, {"t"}, INFINITY, 0}
    };
    vector<string> vars = { "x", "y", "z", "u", "t" };
    double p;
    p = flow_bound(dcs, vars, false);
    assert(abs(p - 1.5) < 1e-7);
    p = flow_bound(dcs, vars, true);
    assert(abs(p - 1.5) < 1e-7);
}

void test_flow_bound5() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 1 },
        { {}, {"y", "z"}, 1, 1 },
        { {}, {"z", "u"}, 1, 1 },
        { {"x", "z"}, {"u"}, INFINITY, 0},
        { {"y", "u"}, {"x"}, INFINITY, 0}
    };
    vector<string> vars = { "x", "y", "z", "u" };
    double p;
    p = flow_bound(dcs, vars, false);
    assert(abs(p - 2) < 1e-7);
    p = flow_bound(dcs, vars, true);
    assert(abs(p - 2) < 1e-7);
}

void test_flow_bound_JOB_Q1() {
    vector<DC<string>> dcs = {
        {{"1"}, {"0MC", "1"}, 1.0, log2(1334883.0)},
        {{"1"}, {"0MC", "1"}, 2.0, log2(1685.8359943956589)},
        {{"1"}, {"0MC", "1"}, 3.0, log2(232.70072156462575)},
        {{"1"}, {"0MC", "1"}, 4.0, log2(111.6218174166884)},
        {{"1"}, {"0MC", "1"}, 5.0, log2(89.39599809855387)},
        {{"1"}, {"0MC", "1"}, 6.0, log2(85.15089958750488)},
        {{"1"}, {"0MC", "1"}, 7.0, log2(84.28626028158547)},
        {{"1"}, {"0MC", "1"}, 8.0, log2(84.08192964838128)},
        {{"1"}, {"0MC", "1"}, 9.0, log2(84.02614781955714)},
        {{"1"}, {"0MC", "1"}, 10.0, log2(84.00904342807583)},
        {{"1"}, {"0MC", "1"}, 11.0, log2(84.00331536944712)},
        {{"1"}, {"0MC", "1"}, 12.0, log2(84.00126786773852)},
        {{"1"}, {"0MC", "1"}, 13.0, log2(84.00050011506285)},
        {{"1"}, {"0MC", "1"}, 14.0, log2(84.00020189112921)},
        {{"1"}, {"0MC", "1"}, 15.0, log2(84.00008295402887)},
        {{"1"}, {"0MC", "1"}, INFINITY, log2(84.0)},
        {{"1"}, {"1", "0T"}, 1.0, log2(2528312.0)},
        {{"1"}, {"1", "0T"}, 2.0, log2(1590.0666652691011)},
        {{"1"}, {"1", "0T"}, 3.0, log2(136.23129614475658)},
        {{"1"}, {"1", "0T"}, 4.0, log2(39.87563999823829)},
        {{"1"}, {"1", "0T"}, 5.0, log2(19.079462387568874)},
        {{"1"}, {"1", "0T"}, 6.0, log2(11.671816317298546)},
        {{"1"}, {"1", "0T"}, 7.0, log2(8.216561214186674)},
        {{"1"}, {"1", "0T"}, 8.0, log2(6.314716145499993)},
        {{"1"}, {"1", "0T"}, 9.0, log2(5.145476861143866)},
        {{"1"}, {"1", "0T"}, 10.0, log2(4.368004394179208)},
        {{"1"}, {"1", "0T"}, 11.0, log2(3.8201068904961626)},
        {{"1"}, {"1", "0T"}, 12.0, log2(3.4164040038172514)},
        {{"1"}, {"1", "0T"}, 13.0, log2(3.108317945962265)},
        {{"1"}, {"1", "0T"}, 14.0, log2(2.8664544674888304)},
        {{"1"}, {"1", "0T"}, 15.0, log2(2.672116432129139)},
        {{"1"}, {"1", "0T"}, 16.0, log2(2.5129098960169647)},
        {{"1"}, {"1", "0T"}, 17.0, log2(2.380329604344474)},
        {{"1"}, {"1", "0T"}, 18.0, log2(2.2683643581100164)},
        {{"1"}, {"1", "0T"}, 19.0, log2(2.17265659251006)},
        {{"1"}, {"1", "0T"}, 20.0, log2(2.089977127668915)},
        {{"1"}, {"1", "0T"}, 21.0, log2(2.0178863305871544)},
        {{"1"}, {"1", "0T"}, 22.0, log2(1.9545093733456904)},
        {{"1"}, {"1", "0T"}, 23.0, log2(1.898383462791728)},
        {{"1"}, {"1", "0T"}, 24.0, log2(1.848351699168005)},
        {{"1"}, {"1", "0T"}, 25.0, log2(1.803487874051152)},
        {{"1"}, {"1", "0T"}, 26.0, log2(1.7630422416840343)},
        {{"1"}, {"1", "0T"}, 27.0, log2(1.7264017855632618)},
        {{"1"}, {"1", "0T"}, 28.0, log2(1.6930606803918251)},
        {{"1"}, {"1", "0T"}, 29.0, log2(1.6625980405997876)},
        {{"1"}, {"1", "0T"}, 30.0, log2(1.6346609532649696)},
        {{"1"}, {"1", "0T"}, INFINITY, log2(1.0)},
        {{"1"}, {"1", "0MI_IDX"}, 1.0, log2(250.0)},
        {{"1"}, {"1", "0MI_IDX"}, 2.0, log2(15.811388300841896)},
        {{"1"}, {"1", "0MI_IDX"}, 3.0, log2(6.299605249474365)},
        {{"1"}, {"1", "0MI_IDX"}, 4.0, log2(3.976353643835253)},
        {{"1"}, {"1", "0MI_IDX"}, 5.0, log2(3.017088168272582)},
        {{"1"}, {"1", "0MI_IDX"}, 6.0, log2(2.509901442183411)},
        {{"1"}, {"1", "0MI_IDX"}, 7.0, log2(2.2007102102809872)},
        {{"1"}, {"1", "0MI_IDX"}, 8.0, log2(1.9940796483178032)},
        {{"1"}, {"1", "0MI_IDX"}, 9.0, log2(1.8468761744797573)},
        {{"1"}, {"1", "0MI_IDX"}, 10.0, log2(1.736976732219687)},
        {{"1"}, {"1", "0MI_IDX"}, 11.0, log2(1.6519410534528962)},
        {{"1"}, {"1", "0MI_IDX"}, 12.0, log2(1.584266846899035)},
        {{"1"}, {"1", "0MI_IDX"}, 13.0, log2(1.5291740650985803)},
        {{"1"}, {"1", "0MI_IDX"}, 14.0, log2(1.4834790899372283)},
        {{"1"}, {"1", "0MI_IDX"}, 15.0, log2(1.4449827655296232)},
        {{"1"}, {"1", "0MI_IDX"}, INFINITY, log2(1.0)}
    };

    vector<string> vars = {"0MC", "0MI_IDX", "0T", "1"};

    double p;
    p = pow(2, flow_bound(dcs, vars, false));
    assert(abs(p-7017) < 1);
    p = pow(2, flow_bound(dcs, vars, true));
    assert(abs(p-7017) < 1);
}



//==========================================================================================
// Unit tests for approximate_topological_sort()
//==========================================================================================

void test_approximate_topological_sort1() {
    vector<string> V = {"A", "B", "C", "D"};
    vector<pair<string, string>> E = {
        {"A", "C"},
        {"A", "D"},
        {"C", "B"},
        {"D", "B"},
    };
    auto p = approximate_topological_sort(V, E);
    assert(p.first == true);
    assert(p.second == vector<string>({"A", "D", "C", "B"}));
}

void test_approximate_topological_sort2() {
    vector<string> V = {"A", "B", "C", "D"};
    vector<pair<string, string>> E = {
        {"A", "D"},
        {"D", "B"},
        {"B", "C"},
    };
    auto p = approximate_topological_sort(V, E);
    assert(p.first == true);
    assert(p.second == vector<string>({"A", "D", "B", "C"}));
}

void test_approximate_topological_sort3() {
    vector<string> V = {"A", "B", "C", "D"};
    vector<pair<string, string>> E = {
        {"A", "B"},
        {"B", "C"},
        {"C", "D"},
        {"D", "A"},
    };
    auto p = approximate_topological_sort(V, E);
    assert(p.first == false);
    assert(p.second == vector<string>({"B", "C", "D", "A"}));
}

void test_approximate_topological_sort4() {
    vector<string> V = {"A", "B", "C", "D"};
    vector<pair<string, string>> E = {
        {"A", "B"},
        {"A", "B"},
        {"B", "C"},
        {"B", "C"},
        {"C", "D"},
        {"C", "D"},
        {"D", "A"},
    };
    auto p = approximate_topological_sort(V, E);
    assert(p.first == false);
    assert(p.second == vector<string>({"A", "B", "C", "D"}));
}

void test_approximate_topological_sort5() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 1 },
        { {}, {"y", "z"}, 1, 1 },
        { {}, {"x", "z"}, 1, 1 },
        { {"x", "z"}, {"u"}, INFINITY, 0},
        { {"y", "z"}, {"t"}, INFINITY, 0}
    };
    vector<string> vars = { "x", "y", "z", "u", "t" };
    auto p = approximate_topological_sort(vars, dcs);
    assert(p.first == true);
    assert(p.second == vector<string>({"z", "y", "x", "u", "t"}));
}

void test_approximate_topological_sort6() {
        vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 1 },
        { {}, {"y", "z"}, 1, 1 },
        { {}, {"z", "u"}, 1, 1 },
        { {"x", "z"}, {"u"}, INFINITY, 0},
        { {"y", "u"}, {"x"}, INFINITY, 0}
    };
    vector<string> vars = { "x", "y", "z", "u" };
    auto p = approximate_topological_sort(vars, dcs);
    assert(p.first == false);
    assert(p.second == vector<string>({"z", "y", "x", "u"}));
}



//==========================================================================================
// Run the test cases
//==========================================================================================

int main() {
    test_lp1();

    test_flow_bound1();
    test_flow_bound2();
    test_flow_bound3();
    test_flow_bound4();
    test_flow_bound5();
    test_flow_bound_JOB_Q1();

    test_approximate_topological_sort1();
    test_approximate_topological_sort2();
    test_approximate_topological_sort3();
    test_approximate_topological_sort4();
    test_approximate_topological_sort5();
    test_approximate_topological_sort6();

    return 0;
}
