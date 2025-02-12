/******************************************************************************************/
// An implementation of the flow bound with Lp-norm constraints on degree sequences.
// The flow bound is a generalization of both:
// - the simple DC bound: given by Equation (1) in [this paper](https://arxiv.org/pdf/2211.08381)
// - the chain bound: described in Section 5.1 in [this paper](https://arxiv.org/pdf/1604.00111)
//
// The ENTRY POINT to this file is the `flow_bound()` function below.
/******************************************************************************************/
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
// The following macro can be used to enable debug output. It should be left UNDEFINED in
// release mode, as it can significantly slow down the code
// #define DEBUG_FLOW_BOUND
/******************************************************************************************/


//==========================================================================================
// Generic interface for building and solving LPs using the HiGHS library
//==========================================================================================

// A variable has a name, lower bound, and upper bound.
struct Variable {
    const string name;
    const double lower_bound;
    const double upper_bound;
};

// A constraint has a name, lower bound, upper bound, and a sum of variables. The sum is
// represented as a dictionary from variable indices to coefficients.
struct Constraint {
    const string name;
    const double lower_bound;
    const double upper_bound;
    unordered_map<int, double> sum;
};

// The objective has a Boolean flag indicating whether it is a maximization or minimization
// and a sum indicating the objective function. The sum is represented as a dictionary from
// variable indices to coefficients.
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

    // Add `a * v` to the constraint `c`
    void add_to_constraint(int c, int v, double a) {
        constraints.at(c).sum[v] += a;
    }

    // Add `a * v` to the objective
    void add_to_objective(int v, double a) {
        objective.sum[v] += a;
    }
};

#ifdef DEBUG_FLOW_BOUND
// Output a variable
ostream& operator<<(ostream& os, const Variable& v) {
    if (v.lower_bound != -INFINITY)
        os << v.lower_bound << " <= ";
    os << v.name;
    if (v.upper_bound != INFINITY)
        os << " <= " << v.upper_bound;
    return os;
}

// Output a sum of variables. The vector `variables` can be used to map variable indices to
// their names
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

// Output a constraint
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

// Output an LP
ostream& operator<<(ostream& os, const LP& lp) {
    os << "Linear Program:" << endl;
    os << (lp.objective.maximize ? "maximize" : "minimize") << " ";
    print_sum(os, lp.objective.sum, lp.variables) << endl;
    os << "subject to" << endl;
    for (const auto& v : lp.variables)
        os << "    " << v << endl;
    for (const auto& c : lp.constraints) {
        os << "    ";
        print_constraint(os, c, lp.variables) << endl;
    }
    return os;
}
#endif

// Convert an LP to a HiGHs LpProblem
pair<double,vector<double>> solve(const LP &p) {
    #ifdef DEBUG_FLOW_BOUND
    cout << p << endl;
    #endif
    int n = p.variables.size();    // Number of variables
    int m = p.constraints.size();  // Number of constraints

    HighsModel model;
    model.lp_.num_col_ = n;
    model.lp_.num_row_ = m;
    // Set the objective
    model.lp_.sense_ = p.objective.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    model.lp_.offset_ = 0.0;
    model.lp_.col_cost_.resize(n);
    for (const auto& v : p.objective.sum)
        model.lp_.col_cost_.at(v.first) = v.second;
    // Set bounds on variables
    model.lp_.col_lower_.resize(n);
    model.lp_.col_upper_.resize(n);
    int i = 0;
    for (const auto& v : p.variables) {
        model.lp_.col_lower_.at(i) = v.lower_bound;
        model.lp_.col_upper_.at(i) = v.upper_bound;
        ++i;
    }
    // Set bounds on constraints
    model.lp_.row_lower_.resize(m);
    model.lp_.row_upper_.resize(m);
    i = 0;
    for (const auto& c : p.constraints) {
        model.lp_.row_lower_.at(i) = c.lower_bound;
        model.lp_.row_upper_.at(i) = c.upper_bound;
        ++i;
    }
    // Set the constraint matrix
    model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
    model.lp_.a_matrix_.start_.resize(m + 1);
    i = 0;
    for (const auto& c : p.constraints) {
        model.lp_.a_matrix_.start_.at(i) = model.lp_.a_matrix_.index_.size();
        for (const auto& t : c.sum) {
            model.lp_.a_matrix_.index_.push_back(t.first);
            model.lp_.a_matrix_.value_.push_back(t.second);
        }
        ++i;
    }
    model.lp_.a_matrix_.start_.at(i) = model.lp_.a_matrix_.index_.size();

    // Create a Highs instance
    Highs highs;
    #ifndef DEBUG_FLOW_BOUND
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

    // TODO: The following corner case is special to the LP corresponding to the flow bound
    if (model_status==HighsModelStatus::kInfeasible)
        return make_pair(INFINITY, vector<double>(n, 0.0));
    if (model_status==HighsModelStatus::kUnbounded)
        return make_pair(p.objective.maximize ? INFINITY : -INFINITY, vector<double>(n, 0.0));

    assert(model_status==HighsModelStatus::kOptimal);

    const HighsInfo& info = highs.getInfo();
    const double obj = info.objective_function_value;
    const bool has_values = info.primal_solution_status;
    assert(has_values);
    const HighsSolution& solution = highs.getSolution();
    assert(solution.col_value.size() == size_t(n));

    // Return a pair of the objective value and the variable values
    return make_pair(obj, solution.col_value);
}

// A testcase for the LP interface
void test_lp1() {
    LP lp(true);
    int x = lp.add_variable("x", 0.0, INFINITY);    // x >= 0
    int y = lp.add_variable("y", 0.0, INFINITY);    // y >= 0
    int z = lp.add_variable("z", 0.0, INFINITY);    // z >= 0
    int t = lp.add_variable("t", 0.0, 2.0);         // 0 <= t <= 2
    // Add a constraint x + y <= 1
    int c1 = lp.add_constraint("c1", -INFINITY, 1.0);
    lp.add_to_constraint(c1, x, 1.0);
    lp.add_to_constraint(c1, y, 1.0);
    // Add a constraint y + z <= 1
    int c2 = lp.add_constraint("c2", -INFINITY, 1.0);
    lp.add_to_constraint(c2, y, 1.0);
    lp.add_to_constraint(c2, z, 1.0);
    int c3 = lp.add_constraint("c3", -INFINITY, 1.0);
    // Add a constraint x + z <= 1
    lp.add_to_constraint(c3, x, 1.0);
    lp.add_to_constraint(c3, z, 1.0);
    // Set the objective to maximize x + y + z + t
    lp.add_to_objective(x, 1.0);
    lp.add_to_objective(y, 1.0);
    lp.add_to_objective(z, 1.0);
    lp.add_to_objective(t, 1.0);
    auto sol = solve(lp);
    auto& obj = sol.first;
    auto& val = sol.second;
    assert(abs(obj - 3.5) < 1e-7);
    // The optimal values are x = y = z = 1/2, t = 2
    assert(abs(val[x] - 0.5) < 1e-7);
    assert(abs(val[y] - 0.5) < 1e-7);
    assert(abs(val[z] - 0.5) < 1e-7);
    assert(abs(val[t] - 2.0) < 1e-7);
}



//==========================================================================================
// Generic utility functions:
//==========================================================================================

// Union of two sets
template <typename T>
set<T> set_union(const set<T>& X, const set<T>& Y) {
    set<T> Z;
    copy(X.begin(), X.end(), inserter(Z, Z.end()));
    copy(Y.begin(), Y.end(), inserter(Z, Z.end()));
    return Z;
}

// Intersection of two sets
template <typename T>
set<T> set_intersection(const set<T>& X, const set<T>& Y) {
    set<T> Z;
    set_intersection(X.begin(), X.end(), Y.begin(), Y.end(), inserter(Z, Z.end()));
    return Z;
}

// Set difference of two sets
template <typename T>
set<T> set_difference(const set<T>& X, const set<T>& Y) {
    set<T> Z;
    set_difference(X.begin(), X.end(), Y.begin(), Y.end(), inserter(Z, Z.end()));
    return Z;
}

// Whether a set is a subset of another
template <typename T>
bool is_subset(const set<T>& X, const set<T>& Y) {
    return includes(Y.begin(), Y.end(), X.begin(), X.end());
}



//==========================================================================================
// The core implementation of the flow bound with Lp-norm constraints
//==========================================================================================

// A representation of a constraint on an Lp-norm of a degree sequence. The constraint is:
// log_2 ||deg(Y|X)||_p <= b
template <typename T>
struct DC {
    // Specify the degree sequence deg(Y|X) using the sets X and Y
    const set<T> X;
    const set<T> Y;
    // Specify the Lp-norm to be used
    const double p;
    // Specify the upper bound on the Lp-norm of the degree sequence *ON LOG SCALE*
    // log_2 ||deg(Y|X)||_p <= b
    const double b;

    DC(const set<T>& X_, const set<T>& Y_, double p_, double b_)
    // Note that this constructor always sets Y to be `X_ union Y_`
    : X(X_), Y(set_union(X_, Y_)), p(p_), b(b_) {
        assert(p > 0.0);
        assert(b >= 0.0);
    }
};

// In Debug mode, we generate meaningful names for variables and constraints.
#ifdef DEBUG_FLOW_BOUND
inline string name(string s) {
    return s;
}

inline string name(int i) {
    return to_string(i);
}

// Given a sorted set `X`, convert each element to a string and concatenate the strings.
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

// Print a degree constraint DC
template <typename T>
ostream& operator<<(ostream& os, const DC<T>& dc) {
    os << "log_2 ||deg(" << set_name(dc.Y) << "|" << set_name(dc.X) << ")||_" <<
        dc.p << " <= " << dc.b;
    return os;
}

// Generate a meaningful name for the variable specifying the `t`-flow from node `X` to node
// `Y`. For context, just like in the simple DC bound, in the flow bound, we have a
// different flow function for every query variable `t`. However, all the flows share the
// same network structure
template <typename T>
inline string flow_var_name(int t, const set<T> &X, const set<T> &Y) {
    return "f" + to_string(t) + "_" + set_name(X) + "->" + set_name(Y);
}

// Generate a meaningful name for the capacity constraint on the `t`-flow from `X` to `Y`
template <typename T>
inline string flow_capacity_con_name(int t, const set<T> &X, const set<T> &Y) {
    return "c" + to_string(t) + "_" + set_name(X) + "->" + set_name(Y);
}

// Generate a meaningful name for the flow conservation constraint on the `t`-flow at node
// `Z`
template <typename T>
inline string flow_conservation_con_name(int t, const set<T> &Z) {
    return "e" + to_string(t) + "_" + set_name(Z);
}

// Generate a meaningful name for the variable specifying the coefficient of the `i`-th DC
inline string dc_var_name(int i) {
    return "a" + to_string(i);
}
// In Release mode, we skip all name generation and just use empty strings everywhere
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

// Given a (weighted) directed multi-graph `G` (i.e. where we can have multiple edges `u ->
// v` between the same pair of vertices `(u, v)`):
//  - if `G` is acyclic, return `true` along with a topological ordering of the vertices.
//  - otherwise, return `false` along an ordering of the vertices that is as close to a
//    topological ordering as possible.
//
// The algorithm is greedy where we keep peeling off the vertex with the smallest weighted
//    outdegree until the graph is empty. The weighted outdegree of a vertex v` is the sum
//    of weights of outgoing edges from `v`
template <typename T>
pair<bool,vector<T>> approximate_topological_sort(
    const vector<T>& V,             // Vertices
    const vector<pair<T, T>>& E,    // Edges
    const vector<double>& W         // Weights
) {
    assert(E.size() == W.size());
    for (auto &w : W)
        assert(w >= 0.0);
    // The weighted outdegree of each vertex
    map<T, double> out_degree;
    // A map from each vertex to the incoming edges along with their weights
    map<T, vector<pair<T, double>>> in_edges;
    for (auto& v : V)
        out_degree[v] = 0.0;
    int i = 0;
    for (auto& e : E) {
        const T& v = e.first;
        const T& u = e.second;
        double w = W[i];
        // Every edge must have both endpoints in V
        assert(out_degree.find(v) != out_degree.end());
        assert(out_degree.find(u) != out_degree.end());
        out_degree[v] += w;
        in_edges[u].push_back(make_pair(v, w));
        ++i;
    }
    // A sorted list of vertices based on their outdegrees
    set<pair<double, T>> Q;
    for (auto& p : out_degree) {
        Q.insert({p.second, p.first});
    }
    bool acyclic = true;
    vector<T> order;
    // Keep peeling off the vertex with the smallest outdegree
    while (!Q.empty()) {
        auto it = Q.begin();
        if (it->first > 1e-6)
            acyclic = false;
        // TODO: Is 1e-6 a good threshold?
        const T& u = it->second;
        order.push_back(u);
        out_degree.erase(u);
        Q.erase(it);
        for (const auto& p : in_edges[u]) {
            const T& v = p.first;
            double w = p.second;
            if (out_degree.find(v) != out_degree.end()) {
                Q.erase({out_degree[v], v});
                out_degree[v] -= w;
                Q.insert({out_degree[v], v});
            }
        }
    }
    reverse(order.begin(), order.end());
    return {acyclic, order};
}

// An overload of the above method where all edges are assigned a weight of 1
template <typename T>
pair<bool,vector<T>> approximate_topological_sort(
    const vector<T>& V,
    const vector<pair<T, T>>& E
) {
    vector<double> W = vector<double>(E.size(), 1.0);
    return approximate_topological_sort(V, E, W);
}


// Instead of a (weighted) directed multi-graph, now we are given a set of DCs. Our goal is
// to translate those DCs into a (weighted) directed multi-graph, and then find an
// approximate topological ordering of the vertices.
//  - `use_weighted_edges` is a Boolean flag that determines whether we want to construct a
//  *weighted* graph or not
template <typename T>
pair<bool,vector<T>> approximate_topological_sort(
    const vector<T>& V, const vector<DC<T>>& dcs, bool use_weighted_edges = true
) {
    vector<pair<T, T>> E;
    vector<double> W;
    // Given a DC `log_2 ||deg(Y|X)||_p <= b`, the "coverage" of this DC is defined as:
    //    coverage := |X| / p + |Y|
    // For every vertex `x` in `X` and every vertex `y` in `Y`, we add a directed edge from
    // `x` to `y`:
    // - If `use_weighted_edges` is false, then we set the weight to be 1.
    // - Otherwise, we set the weight to be:
    //      coverage / b
    //   If `b = 0`, then we use a threshold to prevent division by zero
    for (auto& dc : dcs)
    {
        double w;
        if (!use_weighted_edges)
            w = 1.0;
        else {
            double coverage = dc.X.size() / (double)dc.p + (dc.Y.size() - dc.X.size());
            w = coverage / max(dc.b, 1e-6);
        }
        for (auto& x : dc.X)
            for (auto& y : set_difference(dc.Y, dc.X)) {
                E.push_back({x, y});
                W.push_back(w);
            }
    }
    return approximate_topological_sort(V, E, W);
}

// The core data structure that is used to construct the LP for the flow bound
template <typename T>
struct LpNormLP {
    // The input DCs and target variables
    const vector<DC<T>> dcs;
    const vector<T> target_vars;
    set<T> target_var_set;

    // A flag that determines whether we want to restrict the flow bound to become the
    // weaker chain bound
    bool use_only_chain_bound;
    // The simple and non-simple DCs
    vector<DC<T>> simple_dcs;
    vector<DC<T>> non_simple_dcs;
    // Whether we want to translate the non-simple DCs into a *weighted* graph `G` or not
    bool use_weighted_edges;
    // Whether `G` is acyclic
    bool is_acyclic;
    // An (approximate) topological ordering of the vertices of `G`
    vector<T> var_order;
    // The index of each vertex in the topological ordering
    map<T, int> var_index;

    // The LP for the flow bound
    LP lp;

    // Background: For every query variable `t`, the flow bound has a separate flow
    // function, which we call below the `t`-th flow. However, all the flow functions share
    // the same network architecture.

    // Given an integer `t`, a set `X`, and a set `Y`, `flow_var[(t, X, Y)]` is the variable
    // representing the `t`-th flow from `X` to `Y`.
    map<tuple<int, const set<T>, const set<T>>, int> flow_var;
    // Given an integer `t`, a set `X`, and a set `Y`, `flow_capacity_con[(t, X, Y)]` is the
    // constraint on the capacity of the `t`-th flow from `X` to `Y`.
    map<tuple<int, const set<T>, const set<T>>, int> flow_capacity_con;
    // Given an integer `t` and a set `Z`, `flow_conservation_con[(t, Z)]` is the flow
    // conservation constraint at node `Z` for the `t`-th flow.
    map<tuple<int, const set<T>>, int> flow_conservation_con;
    // Given an integer `i`, `dc_var[i]` is the variable representing the coefficient of the
    // `i`-th DC in the objective function.
    map<int, int> dc_var;

    // `vertices` and `edges` below specify a graph representing the flow network
    // architecture. We construct this graph based on the *simple* DCs. Note that this graph
    // is totally different from the graph `G` that is used to represent the non-simple DCs
    // above.
    set<set<T>> vertices;
    set<pair<set<T>, set<T>>> edges;

    LpNormLP(
        const vector<DC<T>>& dcs_,
        const vector<T>& target_vars_,
        bool use_only_chain_bound_,
        bool use_weighted_edges_
    ) : dcs(dcs_), target_vars(target_vars_), use_only_chain_bound(use_only_chain_bound_),
        use_weighted_edges(use_weighted_edges_), lp(false) {

        for (const auto& v : target_vars) {
            // If the following assertion fails, this means that the input variables
            // `target_vars` contain duplicates
            assert(target_var_set.find(v) == target_var_set.end());

            target_var_set.insert(v);
        }

        // Separate DCs into simple and non-simple
        for (const auto& dc : dcs)
            if (is_simple(dc))
                simple_dcs.push_back(dc);
            else
                non_simple_dcs.push_back(dc);
        // Compute an approximate topological ordering based on the non-simple DCs
        vector<DC<T>> projected_non_simple_dcs;
        for (const auto& dc : non_simple_dcs) {
            auto p = target_projected_dc(dc);
            if (p.first)
                projected_non_simple_dcs.push_back(p.second);
        }
        auto p = approximate_topological_sort(
            target_vars, projected_non_simple_dcs, use_weighted_edges);
        is_acyclic = p.first;
        var_order = p.second;
        for (size_t i = 0; i < var_order.size(); ++i) {
            var_index[var_order[i]] = i;
        }
    }

    // Whether a DC is simple or not
    bool is_simple(const DC<T>& dc) {
        return !use_only_chain_bound && dc.X.size() <= 1;
    }

    // Project the given DC onto the target variables and return the resulting DC
    pair<bool, DC<T>> target_projected_dc(const DC<T>& dc) {
        if (!is_subset(dc.X, target_var_set))
            return {false, DC<T>({}, {}, 1.0, 0.0)};
        return {true, DC<T>(dc.X, set_intersection(dc.Y, target_var_set), dc.p, dc.b)};
    }

    // Given a DC `log_2 ||deg(Y|X)||_p <= b`, return  another DC
    // `log_2 ||deg(Y2|X)||_p <= b`, where `Y2` is the subset of `Y` that only contains
    // those variables that come after `X in the approximate topological ordering `var_order`
    pair<bool, DC<T>> order_consistent_dc(const DC<T>& dc) {
        int last_x = -1;
        for (const auto& x : dc.X) {
            if (var_index.find(x) == var_index.end())
                return {false, DC<T>({}, {}, 1.0, 0.0)};
            last_x = max(last_x, var_index.at(x));
        }
        set<T> new_Y;
        for (const auto& y : dc.Y) {
            if (var_index.find(y) == var_index.end())
                continue;
            if (var_index.at(y) > last_x)
                new_Y.insert(y);
        }
        return {true, DC<T>(dc.X, new_Y, dc.p, dc.b)};
    }

    // Add a flow variable flow_var[(t, X, Y)], which specifies the `t`-th flow from `X` to
    // `Y`.
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

    // Add a flow capacity constraint flow_capacity_con[(t, X, Y)], which specifies the
    // constraint on the capacity of the `t`-th flow from `X` to `Y`.
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

    // Add a flow conservation constraint flow_conservation_con[(t, Z)], which specifies the
    // flow conservation constraint at node `Z` for the `t`-th flow.
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

    // Add a variable representing the coefficient of the `i`-th DC in the objective
    // function.
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

    // In the flow network, add an edge from `X` to `Y` (assuming we don't already have an
    // opposite edge from Y to X)
    void add_edge(const set<T>& X, const set<T>& Y) {
        // For every edge `X -> Y` in the flow network, we must have either X is a subset of
        // Y or Y is a subset of X
        assert(is_subset(X, Y) || is_subset(Y, X));
        if (X != Y && edges.find({Y, X}) == edges.end()) {
            vertices.insert(X);
            vertices.insert(Y);
            edges.insert({X, Y});
        }
    }

    // Construct the flow network based on the *simple* DCs
    void construct_flow_network() {
        for (auto v : target_vars)
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
        for (size_t t = 0; t < target_vars.size(); ++t) {
            // For every vertex `Z`, add a flow conservation constraint
            for (const auto &Z : vertices) {
                if (Z.empty())
                    continue;
                double lower_bound = (Z.size() == 1 && *Z.begin() == target_vars[t]) ?
                    1.0 : 0.0;
                add_flow_conservation_con(t, Z, lower_bound, INFINITY);
            }

            // For every edge `X -> Y`, we must have either X is a subset of Y or Y is a
            // subset of X. We add a flow variable for the `t`-th flow from `X` to `Y`.
            //  - If X is a subset of Y, then the flow variable is unbounded, but we need to
            //    add a capacity constraint.
            //  - If Y is a subset of X, then the flow variable is lower bounded by 0, but
            //    we don't need to add a capacity constraint. (The capacity is infinite in
            //    this case)
            for (const auto &edge : edges) {
                auto &X = edge.first;
                auto &Y = edge.second;
                // If Y is a subset of X, then the flow variable is lower bounded by 0
                double lower_bound = (X.size() <= Y.size()) ? -INFINITY : 0.0;
                double upper_bound = INFINITY;
                int ft_X_Y = add_flow_var(t, X, Y, lower_bound, upper_bound);
                // The flow from X to Y contributes negatively to the flow conservation
                // constraint at `X` and positively to the constraint at `Y`
                if (!X.empty())
                    lp.add_to_constraint(get_flow_conservation_con(t, X), ft_X_Y, -1.0);
                if (!Y.empty())
                    lp.add_to_constraint(get_flow_conservation_con(t, Y), ft_X_Y, 1.0);
                // If X is a subset of Y, add a capacity constraint
                if (X.size() <= Y.size()) {
                    int ct_X_Y = add_flow_capacity_con(t, X, Y, 0.0, INFINITY);
                    lp.add_to_constraint(ct_X_Y, ft_X_Y, -1.0);
                }
            }

            for (size_t i = 0; i < dcs.size(); ++i) {
                int ai = get_dc_var(i);
                // For simple DCs, add coefficients to capacity constraints.
                if (is_simple(dcs[i])) {
                    // Specifically, a simple DC `log_2 ||deg(Y|X)||_p <= b` contributes a
                    // coefficient of 1 to the capacity constraint from X to Y and a
                    // coefficient of 1/p to the capacity constraint from {} to X
                    if (dcs[i].X.size() != dcs[i].Y.size()) {
                        lp.add_to_constraint(
                            get_flow_capacity_con(t, dcs[i].X, dcs[i].Y), ai, 1.0);
                    }
                    if (dcs[i].p != INFINITY && !dcs[i].X.empty()) {
                        lp.add_to_constraint(
                            get_flow_capacity_con(t, {}, dcs[i].X), ai, 1.0 / dcs[i].p);
                    }
                }
                // For non-simple DCs, add coefficients to flow conservation constraints
                else {
                    // Specifically, given a non-simple DC `log_2 ||deg(Y|X)||_p <= b`, let
                    // `log_2 ||deg(Y2|X)||_p <= b` be the version of this DC that is
                    // consistent with the chosen approximate topological ordering
                    // `var_order`. Then, this DC contributes a coefficient of 1 to the flow
                    // conservation constraint for every y in Y2 and a coefficient of 1/p to
                    // the flow conservation constraint for every x in X
                    auto p = order_consistent_dc(dcs[i]);
                    if (!p.first)
                        continue;
                    const DC<T>& dc = p.second;
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

    // Set the objective function to be the sum of the coefficients of the DCs multiplied by
    // their corresponding bounds `dc.b`
    void set_objective() {
        for (size_t i = 0; i < dcs.size(); ++i) {
            lp.add_to_objective(get_dc_var(i), dcs[i].b);
        }
    }
};

// In Release mode, we convert the strings used to represent variables to integers
#ifndef DEBUG_FLOW_BOUND
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
    const vector<DC<string>>& dcs, const vector<string>& target_vars
) {
    set<string> var_set;
    copy(target_vars.begin(), target_vars.end(), inserter(var_set, var_set.end()));
    for (const auto& dc : dcs) {
        copy(dc.X.begin(), dc.X.end(), inserter(var_set, var_set.end()));
        copy(dc.Y.begin(), dc.Y.end(), inserter(var_set, var_set.end()));
    }
    vector<string> vars(var_set.begin(), var_set.end());
    map<string, int> var_map;
    for (size_t i = 0; i < vars.size(); ++i)
        var_map[vars[i]] = i;

    vector<DC<int>> new_dcs;
    for (const auto& dc : dcs)
        new_dcs.push_back(transform_DC(dc, var_map));

    vector<int> new_target_vars;
    for (const auto& v : target_vars)
        new_target_vars.push_back(var_map[v]);

    return make_pair(new_dcs, new_target_vars);
}
#endif

/*************************************************/
// NOTE: This function is the main entry point to this file. It returns a pair `(bound,
// dc_coefs)` where:
//   - `bound` is the flow bound
//   - `dc_coefs` is a vector of the same length as the given `dcs` where `dc_coefs[i]` is
//      the exponent of the Lp-norm corresponding to the `i`-th DC in the flow bound
//
// Example:
// --------
// Suppose we have the triangle query: `Q(X, Y, Z) = R(X, Y), S(Y, Z), T(Z, X)` where the
// L2-norms of the degree sequences `deg_R(Y|X), deg_S(Z|Y), deg_T(X|Z)` are upper bounded
// by a constant C. Then, we have:
//  - `dcs = {DC({"X"}, {"Y"}, 2, log C), DC({"Y"}, {"Z"}, 2, log C), DC({"Z"}, {"X"}, 2, log C)}`
//  - `target_vars = {"X", "Y", "Z"}`
// In this example, `flow_bound(dcs, target_vars)` returns a pair `(bound, dc_coefs)` where:
//  - `bound = 2 log C`
//  - `dc_coefs = {2/3, 2/3, 2/3}`
// The above `bound` and `dc_coefs` correspond to the inequality:
//   |Q| <= [ ||deg_R(Y|X)||_2 * ||deg_S(Z|Y)||_2 * ||deg_T(X|Z)||_2 ]^(2/3)
//       <= C^2
/*************************************************/
pair<double,vector<double>> flow_bound(
    // The degree constraints
    const vector<DC<string>> &dcs,
    // The target set of variables whose cardinality we want to bound.
    // `target_vars` could be a proper subset of the variables in the degree constraints.
    const vector<string> &target_vars,

    //---------------------------------------------------------------------------
    // The following optional parameters are for *TESTING PURPOSES ONLY*. They should be
    // left to their default values in practice.

    // `use_only_chain_bound`: If set to true, the flow bound will be restricted to the
    // chain bound (which is weaker than the flow bound)
    bool use_only_chain_bound = false,
    // `use_weighted_edges`: The flow bound needs to find a variable ordering that is as
    // consistent as possible with the given DCs. By default, it does so by constructing
    // a weighted graph. If `use_weighted_edges` is set to false, it will ignore the weights
    // resulting in a weaker heuristic.
    bool use_weighted_edges = true
    //---------------------------------------------------------------------------
) {
    // In debug mode, we operate directly on the given strings used to represent variables
    #ifdef DEBUG_FLOW_BOUND
    cout << "Degree Constraints:" << endl;
    for (const auto& dc : dcs) {
        cout << "    " << dc << endl;
    }
    cout << endl;
    LpNormLP<string> lp(dcs, target_vars, use_only_chain_bound, use_weighted_edges);
    #else
    // In release mode, we convert the strings to numbers
    auto int_dcs_vars = transform_dcs_to_int(dcs, target_vars);
    LpNormLP<int> lp(
        int_dcs_vars.first, int_dcs_vars.second, use_only_chain_bound, use_weighted_edges);
    #endif
    lp.construct_flow_network();
    lp.add_flow_constraints();
    lp.set_objective();
    auto sol = solve(lp.lp);
    // Extract the coefficients of the DCs from the solution
    vector<double> dc_coefs;
    for (size_t i = 0; i < dcs.size(); ++i)
        dc_coefs.push_back(sol.second.at(lp.get_dc_var(i)));
    return {sol.first, dc_coefs};
}



//==========================================================================================
// Lp-norm bound based on (elemental) Shannon inequalities
//==========================================================================================

template <typename T>
struct ShannonLP {
    // The input DCs and target variables
    const vector<DC<T>> dcs;
    const vector<T> target_vars;

    vector<T> vars;
    map<T,int> var_map;

    LP lp;

    ShannonLP(const vector<DC<T>>& dcs_, const vector<T>& target_vars_)
        : dcs(dcs_), target_vars(target_vars_), lp(true) {
        set<string> var_set;
        copy(target_vars.begin(), target_vars.end(), inserter(var_set, var_set.end()));
        for (const auto& dc : dcs) {
            copy(dc.X.begin(), dc.X.end(), inserter(var_set, var_set.end()));
            copy(dc.Y.begin(), dc.Y.end(), inserter(var_set, var_set.end()));
        }
        copy(var_set.begin(), var_set.end(), back_inserter(vars));
        for (size_t i = 0; i < vars.size(); ++i)
            var_map[vars[i]] = i;
    }

    int zip(const set<T>& U) {
        int z = 0;
        for (const auto& u : U)
            z |= 1 << var_map.at(u);
        return z;
    }

    set<T> unzip(int z) {
        set<T> U;
        int i = 0;
        while (z != 0) {
            if ((z & 1) != 0)
                U.insert(vars[i]);
            z >>= 1;
            ++i;
        }
        return U;
    }

    string name(int z) {
        set<T> U = unzip(z);
        string n = "h(";
        bool first = true;
        for (const auto& u : U) {
            if (first)
                first = false;
            else
                n += ",";
            n += u;
        }
        return n + ")";
    }

    void construct_lp() {
        int n = vars.size();
        int N = 1 << n;

        lp.add_variable("h()", 0.0, 0.0);
        for (int i = 1; i < N; ++i)
            lp.add_variable(name(i), -INFINITY, INFINITY);

        int Y = N - 1;
        for (int y = 0; y < n; ++y){
            int X = Y & ~(1 << y);
            int mon_con = lp.add_constraint("monotonicity", 0.0, INFINITY);
            lp.add_to_constraint(mon_con, Y, 1.0);
            lp.add_to_constraint(mon_con, X, -1.0);
        }

        for (int X = 0; X < N; ++X)
            for (int y = 0; y < n; ++y)
                for (int z = y + 1; z < n; ++z) {
                    if (((X & (1 << y)) != 0) || ((X & (1 << z)) != 0))
                        continue;
                    int Y = X | (1 << y);
                    int Z = X | (1 << z);
                    int W = Y | (1 << z);
                    int sub_con = lp.add_constraint("submodularity", 0.0, INFINITY);
                    lp.add_to_constraint(sub_con, Y, 1.0);
                    lp.add_to_constraint(sub_con, Z, 1.0);
                    lp.add_to_constraint(sub_con, X, -1.0);
                    lp.add_to_constraint(sub_con, W, -1.0);
                }

        for (const auto& dc : dcs) {
            int X = zip(dc.X);
            int Y = zip(dc.Y);
            int con = lp.add_constraint("DC", -INFINITY, dc.b);
            lp.add_to_constraint(con, Y, 1.0);
            double cX = -1.0 + 1.0 / dc.p;
            if (X != 0 && cX != 0.0)
                lp.add_to_constraint(con, X, cX);
        }

        int X = zip(set<T>(target_vars.begin(), target_vars.end()));
        lp.add_to_objective(X, 1.0);
    }
};

// The Lp-norm bound based on (elemental) Shannon inequalities
double elemental_shannon_bound(
    const vector<DC<string>> &dcs,
    const vector<string> &target_vars
) {
    ShannonLP<string> lp(dcs, target_vars);
    lp.construct_lp();
    #ifdef DEBUG_FLOW_BOUND
    cout << lp.lp << endl;
    #endif
    return solve(lp.lp).first;
}

//==========================================================================================
// Testcases for the flow_bound function
//==========================================================================================

// Check whether two vectors are approximately equal
bool are_approx_equal(const vector<double>& x, const vector<double>& y, double eps = 1e-7) {
    if (x.size() != y.size())
        return false;
    for (size_t i = 0; i < x.size(); ++i)
        if (abs(x[i] - y[i]) > eps)
            return false;
    return true;
}

void test_flow_bound1() {
    vector<DC<string>> dcs = {
        { {}, {"A", "B"}, 1, 10 },
        { {}, {"A", "C"}, 1, 10 },
        { {}, {"B", "C"}, 1, 10 }
    };
    vector<string> vars = { "A", "B", "C" };
    pair<double,vector<double>> sol;
    sol = flow_bound(dcs, vars, false);
    assert(abs(sol.first - 15) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({0.5, 0.5, 0.5})));
    sol = flow_bound(dcs, vars, true);
    assert(abs(sol.first - 15) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({0.5, 0.5, 0.5})));

    double p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 15) < 1e-7);
}

void test_flow_bound1a() {
    vector<DC<string>> dcs = {
        { {}, {"A", "B"}, 1, 5 },
        { {}, {"A", "C"}, 1, 20 },
        { {}, {"B", "C"}, 1, 5 }
    };
    vector<string> vars = { "A", "B", "C" };
    pair<double,vector<double>> sol;
    sol = flow_bound(dcs, vars, false);
    assert(abs(sol.first - 10) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({1.0, 0.0, 1.0})));
    sol = flow_bound(dcs, vars, true);
    assert(abs(sol.first - 10) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({1.0, 0.0, 1.0})));

    double p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 10) < 1e-7);
}

void test_flow_bound2() {
    vector<DC<string>> dcs = {
        { {}, {"A", "B"}, INFINITY, 1 },
        { {}, {"A", "C"}, INFINITY, 1 },
        { {}, {"B", "C"}, INFINITY, 1 }
    };
    vector<string> vars = { "A", "B", "C" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 1.5) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(abs(p - 1.5) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 1.5) < 1e-7);
}

void test_flow_bound3() {
    vector<DC<string>> dcs = {
        { {"A"}, {"B"}, 2, 10 },
        { {"B"}, {"C"}, 2, 10 },
        { {"C"}, {"A"}, 2, 10 }
    };
    vector<string> vars = { "A", "B", "C" };
    pair<double,vector<double>> sol;
    sol = flow_bound(dcs, vars, false);
    assert(abs(sol.first - 20) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({2/3.0, 2/3.0, 2/3.0})));
    sol = flow_bound(dcs, vars, true);
    assert(abs(sol.first - 30) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({0.0, 2.0, 1.0})));

    double p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 20) < 1e-7);
}

void test_flow_bound4() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 10 },
        { {}, {"y", "z"}, 1, 10 },
        { {}, {"x", "z"}, 1, 10 },
        { {"x", "z"}, {"u"}, INFINITY, 0},
        { {"y", "z"}, {"t"}, INFINITY, 0}
    };
    vector<string> vars = { "x", "y", "z", "u", "t" };
    pair<double,vector<double>> sol;
    sol = flow_bound(dcs, vars, false);
    assert(abs(sol.first - 15) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({0.5, 0.5, 0.5, 1.0, 1.0})));
    sol = flow_bound(dcs, vars, true);
    assert(abs(sol.first - 15) < 1e-7);
    assert(are_approx_equal(sol.second, vector<double>({0.5, 0.5, 0.5, 1.0, 1.0})));

    double p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 15) < 1e-7);
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
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 2) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(abs(p - 2) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 1.5) < 1e-7);
}

void test_flow_bound6() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 10 },
        { {}, {"y", "z"}, 1, 10 },
        { {}, {"x", "z"}, 1, 10 },
        { {"x", "y"}, {"u"}, 4, 1},
        { {"y", "z"}, {"u"}, 4, 1},
        { {"x", "z"}, {"u"}, 4, 1},
    };
    vector<string> vars = { "x", "y", "z", "u" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 6) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(abs(p - 6) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 6) < 1e-7);
}

void test_flow_bound7() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 10 },
        { {}, {"y", "z"}, 1, 10 },
        { {}, {"x", "z"}, 1, 10 },
        { {"x", "y"}, {"u"}, 4, 1},
        { {"y", "z"}, {"u"}, 4, 1},
        { {"x", "z"}, {"u"}, 4, 1},
        { {"u"}, {"x"}, INFINITY, 100},
        { {"u"}, {"x"}, INFINITY, 100},
        { {"u"}, {"x"}, INFINITY, 100},
        { {"u"}, {"y"}, INFINITY, 100},
        { {"u"}, {"y"}, INFINITY, 100},
        { {"u"}, {"y"}, INFINITY, 100},
        { {"u"}, {"z"}, INFINITY, 100},
        { {"u"}, {"z"}, INFINITY, 100},
        { {"u"}, {"z"}, INFINITY, 100},
    };
    vector<string> vars = { "x", "y", "z", "u" };
    double p;
    p = flow_bound(dcs, vars, false, true).first;
    assert(abs(p - 6) < 1e-7);
    p = flow_bound(dcs, vars, false, false).first;
    assert(abs(p - 6) < 1e-7);
    p = flow_bound(dcs, vars, true, false).first;
    assert(isinf(p) && p > 0);
    p = flow_bound(dcs, vars, true, true).first;
    assert(abs(p - 6) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 6) < 1e-7);
}

void test_flow_bound8() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 2 },
        { {}, {"y", "z"}, 1, 2 },
        { {}, {"x", "z"}, 1, 2 },
        { {"x", "y"}, {"u"}, 4, 1},
        { {"y", "z"}, {"u"}, 4, 1},
        { {"x", "z"}, {"u"}, 4, 1},
    };
    vector<string> vars = { "x", "y", "z", "u" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 3.5) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(abs(p - 3.5) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 3.5) < 1e-7);
}

void test_flow_bound9() {
    vector<DC<string>> dcs = {
        { {"x"}, {"y"}, 3, 2.7 },
        { {"y"}, {"z"}, 3, 2.7 },
        { {"z"}, {"t"}, 3, 2.7 },
        { {"t"}, {"x"}, 3, 2.7 },
    };
    vector<string> vars = { "x", "y", "z", "t" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 2.7 * 3) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(abs(p - 2.7 * (4 + 2.0/3.0)) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 2.7 * 3) < 1e-7);
}

void test_flow_bound10() {
    vector<DC<string>> dcs = {
        { {"x"}, {"y"}, 4, 2.7 },
        { {"y"}, {"z"}, 4, 2.7 },
        { {"z"}, {"t"}, 4, 2.7 },
        { {"t"}, {"x"}, 4, 2.7 },
    };
    vector<string> vars = { "x", "y", "z", "t" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 2.7 * 4) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(abs(p - 2.7 * 5.75) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 2.7 * 4) < 1e-7);
}

void test_flow_bound_infeasible() {
    vector<DC<string>> dcs = {
        { {"x"}, {"y"}, INFINITY, 0 },
        { {"y"}, {"z"}, INFINITY, 0 },
        { {"z"}, {"t"}, INFINITY, 0 },
        { {"t"}, {"x"}, INFINITY, 0 },
    };
    vector<string> vars = { "x", "y", "z", "t" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(isinf(p) && p > 0);
    p = flow_bound(dcs, vars, true).first;
    assert(isinf(p) && p > 0);

    p = elemental_shannon_bound(dcs, vars);
    assert(isinf(p) && p > 0);
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
    p = pow(2, flow_bound(dcs, vars, false).first);
    assert(abs(p-7017) < 1);
    p = pow(2, flow_bound(dcs, vars, true).first);
    assert(abs(p-7017) < 1);

    p = pow(2, elemental_shannon_bound(dcs, vars));
    assert(abs(p-7017) < 1);
}

void test_flow_bound_projection1() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 1 },
        { {}, {"x", "z"}, 1, 1 },
        { {}, {"x", "t"}, 1, 1 },
        { {}, {"y", "z"}, 1, 1 },
        { {}, {"y", "t"}, 1, 1 },
        { {}, {"z", "t"}, 1, 1 },
    };
    vector<string> vars = { "y", "z", "t" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 1.5) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(abs(p - 1.5) < 1e-7);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 1.5) < 1e-7);
}

void test_flow_bound_projection2() {
    vector<DC<string>> dcs = {
        { {}, {"x"}, 1, 10 },
        { {"x"}, {"y"}, INFINITY, 1 },
        { {"y"}, {"z"}, INFINITY, 1 },
    };
    vector<string> vars = { "x", "z" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 12) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(isinf(p) && p > 0);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 12) < 1e-7);
}

void test_flow_bound_projection3() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 10 },
        { {"x", "y"}, {"z"}, INFINITY, 1 },
        { {"y", "z"}, {"t"}, INFINITY, 1 },
        { {"x"}, {"z"}, INFINITY, 10},
        { {"z"}, {"t"}, INFINITY, 10},
    };
    vector<string> vars = { "x", "y", "t" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(abs(p - 30) < 1e-7);
    p = flow_bound(dcs, vars, true).first;
    assert(isinf(p) && p > 0);

    p = elemental_shannon_bound(dcs, vars);
    assert(abs(p - 12) < 1e-7);
}

void test_flow_bound_projection4() {
    vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 10 },
        { {"x", "y"}, {"z", "t"}, INFINITY, 1 },
    };
    vector<string> vars = { "x", "w" };
    double p;
    p = flow_bound(dcs, vars, false).first;
    assert(isinf(p) && p > 0);

    p = elemental_shannon_bound(dcs, vars);
    assert(isinf(p) && p > 0);

    vector<string> vars2 = {"x", "y", "t"};
    p = flow_bound(dcs, vars2, false).first;
    assert(abs(p - 11) < 1e-7);

    p = elemental_shannon_bound(dcs, vars2);
    assert(abs(p - 11) < 1e-7);
}

void test_flow_bound_job_join_1(){
    vector<DC<string>> dcs = {
        {{}, {"0CT", "1"}, 1, 2.0},
        {{"1"}, {"0CT", "1"}, 2, 1.0},
        {{"1"}, {"0CT", "1"}, 3, 0.6666666666666666},
        {{"1"}, {"0CT", "1"}, 4, 0.5},
        {{"1"}, {"0CT", "1"}, 5, 0.4},
        {{"1"}, {"0CT", "1"}, 6, 0.3333333333333333},
        {{"1"}, {"0CT", "1"}, 7, 0.2857142857142857},
        {{"1"}, {"0CT", "1"}, 8, 0.25},
        {{"1"}, {"0CT", "1"}, 9, 0.2222222222222222},
        {{"1"}, {"0CT", "1"}, 10, 0.2},
        {{"1"}, {"0CT", "1"}, 11, 0.18181818181818182},
        {{"1"}, {"0CT", "1"}, 12, 0.16666666666666666},
        {{"1"}, {"0CT", "1"}, 13, 0.15384615384615385},
        {{"1"}, {"0CT", "1"}, 14, 0.14285714285714285},
        {{"1"}, {"0CT", "1"}, 15, 0.13333333333333333},
        {{"1"}, {"0CT", "1"}, 16, 0.125},
        {{"1"}, {"0CT", "1"}, 17, 0.11764705882352941},
        {{"1"}, {"0CT", "1"}, 18, 0.1111111111111111},
        {{"1"}, {"0CT", "1"}, 19, 0.10526315789473684},
        {{"1"}, {"0CT", "1"}, 20, 0.1},
        {{"1"}, {"0CT", "1"}, 21, 0.09523809523809523},
        {{"1"}, {"0CT", "1"}, 22, 0.09090909090909091},
        {{"1"}, {"0CT", "1"}, 23, 0.08695652173913043},
        {{"1"}, {"0CT", "1"}, 24, 0.08333333333333333},
        {{"1"}, {"0CT", "1"}, 25, 0.08},
        {{"1"}, {"0CT", "1"}, 26, 0.07692307692307693},
        {{"1"}, {"0CT", "1"}, 27, 0.07407407407407407},
        {{"1"}, {"0CT", "1"}, 28, 0.07142857142857142},
        {{"1"}, {"0CT", "1"}, 29, 0.06896551724137931},
        {{"1"}, {"0CT", "1"}, 30, 0.06666666666666667},
        {{"1"}, {"0CT", "1"}, INFINITY, 0.0},
        {{}, {"0IT", "3"}, 1, 6.82018},
        {{"3"}, {"0IT", "3"}, 2, 3.41009},
        {{"3"}, {"0IT", "3"}, 3, 2.2733933333333334},
        {{"3"}, {"0IT", "3"}, 4, 1.705045},
        {{"3"}, {"0IT", "3"}, 5, 1.364036},
        {{"3"}, {"0IT", "3"}, 6, 1.1366966666666667},
        {{"3"}, {"0IT", "3"}, 7, 0.9743114285714285},
        {{"3"}, {"0IT", "3"}, 8, 0.8525225},
        {{"3"}, {"0IT", "3"}, 9, 0.7577977777777778},
        {{"3"}, {"0IT", "3"}, 10, 0.682018},
        {{"3"}, {"0IT", "3"}, 11, 0.6200163636363636},
        {{"3"}, {"0IT", "3"}, 12, 0.5683483333333333},
        {{"3"}, {"0IT", "3"}, 13, 0.5246292307692307},
        {{"3"}, {"0IT", "3"}, 14, 0.48715571428571425},
        {{"3"}, {"0IT", "3"}, 15, 0.4546786666666666},
        {{"3"}, {"0IT", "3"}, 16, 0.42626125},
        {{"3"}, {"0IT", "3"}, 17, 0.4011870588235294},
        {{"3"}, {"0IT", "3"}, 18, 0.3788988888888889},
        {{"3"}, {"0IT", "3"}, 19, 0.35895684210526313},
        {{"3"}, {"0IT", "3"}, 20, 0.341009},
        {{"3"}, {"0IT", "3"}, 21, 0.32477047619047617},
        {{"3"}, {"0IT", "3"}, 22, 0.3100081818181818},
        {{"3"}, {"0IT", "3"}, 23, 0.29652956521739127},
        {{"3"}, {"0IT", "3"}, 24, 0.28417416666666667},
        {{"3"}, {"0IT", "3"}, 25, 0.27280719999999997},
        {{"3"}, {"0IT", "3"}, 26, 0.26231461538461537},
        {{"3"}, {"0IT", "3"}, 27, 0.25259925925925925},
        {{"3"}, {"0IT", "3"}, 28, 0.24357785714285712},
        {{"3"}, {"0IT", "3"}, 29, 0.23517862068965517},
        {{"3"}, {"0IT", "3"}, 30, 0.2273393333333333},
        {{"3"}, {"0IT", "3"}, INFINITY, 0.0},
        {{}, {"0MC", "1", "2"}, 1, 21.3151},
        {{"2"}, {"0MC", "1", "2"}, 2, 11.88965},
        {{"2"}, {"0MC", "1", "2"}, 3, 9.249933333333333},
        {{"2"}, {"0MC", "1", "2"}, 4, 8.149925},
        {{"2"}, {"0MC", "1", "2"}, 5, 7.57144},
        {{"2"}, {"0MC", "1", "2"}, 6, 7.227833333333333},
        {{"2"}, {"0MC", "1", "2"}, 7, 7.010671428571428},
        {{"2"}, {"0MC", "1", "2"}, 8, 6.8689875},
        {{"2"}, {"0MC", "1", "2"}, 9, 6.775055555555555},
        {{"2"}, {"0MC", "1", "2"}, 10, 6.71216},
        {{"2"}, {"0MC", "1", "2"}, 11, 6.669636363636363},
        {{"2"}, {"0MC", "1", "2"}, 12, 6.640525},
        {{"2"}, {"0MC", "1", "2"}, 13, 6.620276923076923},
        {{"2"}, {"0MC", "1", "2"}, 14, 6.605914285714285},
        {{"2"}, {"0MC", "1", "2"}, 15, 6.595513333333333},
        {{"2"}, {"0MC", "1", "2"}, 16, 6.5878125},
        {{"2"}, {"0MC", "1", "2"}, 17, 6.582000000000001},
        {{"2"}, {"0MC", "1", "2"}, 18, 6.5775},
        {{"2"}, {"0MC", "1", "2"}, 19, 6.573947368421052},
        {{"2"}, {"0MC", "1", "2"}, 20, 6.57115},
        {{"2"}, {"0MC", "1", "2"}, 21, 6.5688571428571425},
        {{"2"}, {"0MC", "1", "2"}, 22, 6.566954545454546},
        {{"2"}, {"0MC", "1", "2"}, 23, 6.565391304347826},
        {{"2"}, {"0MC", "1", "2"}, 24, 6.564041666666667},
        {{"2"}, {"0MC", "1", "2"}, 25, 6.56292},
        {{"2"}, {"0MC", "1", "2"}, 26, 6.561961538461538},
        {{"2"}, {"0MC", "1", "2"}, 27, 6.561111111111111},
        {{"2"}, {"0MC", "1", "2"}, 28, 6.560392857142857},
        {{"2"}, {"0MC", "1", "2"}, 29, 6.559758620689656},
        {{"2"}, {"0MC", "1", "2"}, 30, 6.559200000000001},
        {{"2"}, {"0MC", "1", "2"}, INFINITY, 6.55459},
        {{}, {"0MC", "1", "2"}, 1, 21.3151},
        {{"1"}, {"0MC", "1", "2"}, 2, 20.81555},
        {{"1"}, {"0MC", "1", "2"}, 3, 20.64923333333333},
        {{"1"}, {"0MC", "1", "2"}, 4, 20.5663},
        {{"1"}, {"0MC", "1", "2"}, 5, 20.5166},
        {{"1"}, {"0MC", "1", "2"}, 6, 20.483666666666668},
        {{"1"}, {"0MC", "1", "2"}, 7, 20.460285714285714},
        {{"1"}, {"0MC", "1", "2"}, 8, 20.442875},
        {{"1"}, {"0MC", "1", "2"}, 9, 20.429333333333332},
        {{"1"}, {"0MC", "1", "2"}, 10, 20.4186},
        {{"1"}, {"0MC", "1", "2"}, 11, 20.40990909090909},
        {{"1"}, {"0MC", "1", "2"}, 12, 20.402666666666665},
        {{"1"}, {"0MC", "1", "2"}, 13, 20.396692307692305},
        {{"1"}, {"0MC", "1", "2"}, 14, 20.39157142857143},
        {{"1"}, {"0MC", "1", "2"}, 15, 20.387133333333335},
        {{"1"}, {"0MC", "1", "2"}, 16, 20.383375},
        {{"1"}, {"0MC", "1", "2"}, 17, 20.380058823529414},
        {{"1"}, {"0MC", "1", "2"}, 18, 20.377111111111113},
        {{"1"}, {"0MC", "1", "2"}, 19, 20.374578947368423},
        {{"1"}, {"0MC", "1", "2"}, 20, 20.372300000000003},
        {{"1"}, {"0MC", "1", "2"}, 21, 20.370238095238093},
        {{"1"}, {"0MC", "1", "2"}, 22, 20.368409090909093},
        {{"1"}, {"0MC", "1", "2"}, 23, 20.36678260869565},
        {{"1"}, {"0MC", "1", "2"}, 24, 20.365333333333332},
        {{"1"}, {"0MC", "1", "2"}, 25, 20.364},
        {{"1"}, {"0MC", "1", "2"}, 26, 20.36276923076923},
        {{"1"}, {"0MC", "1", "2"}, 27, 20.361666666666665},
        {{"1"}, {"0MC", "1", "2"}, 28, 20.360678571428572},
        {{"1"}, {"0MC", "1", "2"}, 29, 20.359758620689654},
        {{"1"}, {"0MC", "1", "2"}, 30, 20.358933333333333},
        {{"1"}, {"0MC", "1", "2"}, INFINITY, 20.3483},
        {{}, {"0MI.IDX", "2", "3"}, 1, 20.3963},
        {{"2"}, {"0MI.IDX", "2", "3"}, 2, 10.9908},
        {{"2"}, {"0MI.IDX", "2", "3"}, 3, 7.855666666666667},
        {{"2"}, {"0MI.IDX", "2", "3"}, 4, 6.288175},
        {{"2"}, {"0MI.IDX", "2", "3"}, 5, 5.3477},
        {{"2"}, {"0MI.IDX", "2", "3"}, 6, 4.720766666666667},
        {{"2"}, {"0MI.IDX", "2", "3"}, 7, 4.273014285714286},
        {{"2"}, {"0MI.IDX", "2", "3"}, 8, 3.9372625},
        {{"2"}, {"0MI.IDX", "2", "3"}, 9, 3.6761888888888894},
        {{"2"}, {"0MI.IDX", "2", "3"}, 10, 3.4674300000000002},
        {{"2"}, {"0MI.IDX", "2", "3"}, 11, 3.296727272727273},
        {{"2"}, {"0MI.IDX", "2", "3"}, 12, 3.1546083333333335},
        {{"2"}, {"0MI.IDX", "2", "3"}, 13, 3.0345153846153847},
        {{"2"}, {"0MI.IDX", "2", "3"}, 14, 2.931771428571429},
        {{"2"}, {"0MI.IDX", "2", "3"}, 15, 2.842966666666667},
        {{"2"}, {"0MI.IDX", "2", "3"}, 16, 2.76555},
        {{"2"}, {"0MI.IDX", "2", "3"}, 17, 2.697605882352941},
        {{"2"}, {"0MI.IDX", "2", "3"}, 18, 2.6376388888888886},
        {{"2"}, {"0MI.IDX", "2", "3"}, 19, 2.584505263157895},
        {{"2"}, {"0MI.IDX", "2", "3"}, 20, 2.537315},
        {{"2"}, {"0MI.IDX", "2", "3"}, 21, 2.495342857142857},
        {{"2"}, {"0MI.IDX", "2", "3"}, 22, 2.4580363636363636},
        {{"2"}, {"0MI.IDX", "2", "3"}, 23, 2.424917391304348},
        {{"2"}, {"0MI.IDX", "2", "3"}, 24, 2.3955958333333336},
        {{"2"}, {"0MI.IDX", "2", "3"}, 25, 2.36972},
        {{"2"}, {"0MI.IDX", "2", "3"}, 26, 2.346953846153846},
        {{"2"}, {"0MI.IDX", "2", "3"}, 27, 2.326977777777778},
        {{"2"}, {"0MI.IDX", "2", "3"}, 28, 2.309467857142857},
        {{"2"}, {"0MI.IDX", "2", "3"}, 29, 2.2941103448275864},
        {{"2"}, {"0MI.IDX", "2", "3"}, 30, 2.2806066666666664},
        {{"2"}, {"0MI.IDX", "2", "3"}, INFINITY, 2.0},
        {{}, {"0MI.IDX", "2", "3"}, 1, 20.3963},
        {{"3"}, {"0MI.IDX", "2", "3"}, 2, 19.6035},
        {{"3"}, {"0MI.IDX", "2", "3"}, 3, 19.339366666666667},
        {{"3"}, {"0MI.IDX", "2", "3"}, 4, 19.207275},
        {{"3"}, {"0MI.IDX", "2", "3"}, 5, 19.12804},
        {{"3"}, {"0MI.IDX", "2", "3"}, 6, 19.075166666666664},
        {{"3"}, {"0MI.IDX", "2", "3"}, 7, 19.03742857142857},
        {{"3"}, {"0MI.IDX", "2", "3"}, 8, 19.009125},
        {{"3"}, {"0MI.IDX", "2", "3"}, 9, 18.98711111111111},
        {{"3"}, {"0MI.IDX", "2", "3"}, 10, 18.9695},
        {{"3"}, {"0MI.IDX", "2", "3"}, 11, 18.95509090909091},
        {{"3"}, {"0MI.IDX", "2", "3"}, 12, 18.943083333333334},
        {{"3"}, {"0MI.IDX", "2", "3"}, 13, 18.932923076923075},
        {{"3"}, {"0MI.IDX", "2", "3"}, 14, 18.924285714285713},
        {{"3"}, {"0MI.IDX", "2", "3"}, 15, 18.916733333333333},
        {{"3"}, {"0MI.IDX", "2", "3"}, 16, 18.910125},
        {{"3"}, {"0MI.IDX", "2", "3"}, 17, 18.90429411764706},
        {{"3"}, {"0MI.IDX", "2", "3"}, 18, 18.89911111111111},
        {{"3"}, {"0MI.IDX", "2", "3"}, 19, 18.894473684210528},
        {{"3"}, {"0MI.IDX", "2", "3"}, 20, 18.8903},
        {{"3"}, {"0MI.IDX", "2", "3"}, 21, 18.886523809523812},
        {{"3"}, {"0MI.IDX", "2", "3"}, 22, 18.88309090909091},
        {{"3"}, {"0MI.IDX", "2", "3"}, 23, 18.87995652173913},
        {{"3"}, {"0MI.IDX", "2", "3"}, 24, 18.877083333333335},
        {{"3"}, {"0MI.IDX", "2", "3"}, 25, 18.87444},
        {{"3"}, {"0MI.IDX", "2", "3"}, 26, 18.872},
        {{"3"}, {"0MI.IDX", "2", "3"}, 27, 18.869740740740742},
        {{"3"}, {"0MI.IDX", "2", "3"}, 28, 18.867642857142858},
        {{"3"}, {"0MI.IDX", "2", "3"}, 29, 18.865689655172414},
        {{"3"}, {"0MI.IDX", "2", "3"}, 30, 18.86386666666667},
        {{"3"}, {"0MI.IDX", "2", "3"}, INFINITY, 18.811},
        {{}, {"0T", "2"}, 1, 21.2697},
        {{"2"}, {"0T", "2"}, 2, 10.63485},
        {{"2"}, {"0T", "2"}, 3, 7.0899},
        {{"2"}, {"0T", "2"}, 4, 5.317425},
        {{"2"}, {"0T", "2"}, 5, 4.25394},
        {{"2"}, {"0T", "2"}, 6, 3.54495},
        {{"2"}, {"0T", "2"}, 7, 3.0385285714285715},
        {{"2"}, {"0T", "2"}, 8, 2.6587125},
        {{"2"}, {"0T", "2"}, 9, 2.3633},
        {{"2"}, {"0T", "2"}, 10, 2.12697},
        {{"2"}, {"0T", "2"}, 11, 1.9336090909090908},
        {{"2"}, {"0T", "2"}, 12, 1.772475},
        {{"2"}, {"0T", "2"}, 13, 1.6361307692307692},
        {{"2"}, {"0T", "2"}, 14, 1.5192642857142857},
        {{"2"}, {"0T", "2"}, 15, 1.41798},
        {{"2"}, {"0T", "2"}, 16, 1.32935625},
        {{"2"}, {"0T", "2"}, 17, 1.2511588235294118},
        {{"2"}, {"0T", "2"}, 18, 1.18165},
        {{"2"}, {"0T", "2"}, 19, 1.119457894736842},
        {{"2"}, {"0T", "2"}, 20, 1.063485},
        {{"2"}, {"0T", "2"}, 21, 1.0128428571428572},
        {{"2"}, {"0T", "2"}, 22, 0.9668045454545454},
        {{"2"}, {"0T", "2"}, 23, 0.9247695652173913},
        {{"2"}, {"0T", "2"}, 24, 0.8862375},
        {{"2"}, {"0T", "2"}, 25, 0.850788},
        {{"2"}, {"0T", "2"}, 26, 0.8180653846153846},
        {{"2"}, {"0T", "2"}, 27, 0.7877666666666667},
        {{"2"}, {"0T", "2"}, 28, 0.7596321428571429},
        {{"2"}, {"0T", "2"}, 29, 0.7334379310344827},
        {{"2"}, {"0T", "2"}, 30, 0.70899},
        {{"2"}, {"0T", "2"}, INFINITY, 0.0}
    };

    vector<string> vars = {"0CT", "0IT", "0MC", "0MI.IDX", "0T", "1", "2", "3" };

    double p1 = flow_bound(dcs, vars).first;
    assert(abs(p1 - 22.8804) < 1e-4);

    double p2 = elemental_shannon_bound(dcs, vars);
    assert(abs(p2 - 22.8804) < 1e-4);

    assert(abs(p1 - p2) < 1e-7);
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

    vector<double> W2 = {10.0, 1.0, 10.0, 10.0};
    auto p2 = approximate_topological_sort(V, E, W2);
    assert(p2.first == false);
    assert(p2.second == vector<string>({"C", "D", "A", "B"}));

    vector<double> W3 = {10.0, 10.0, 1.0, 10.0};
    auto p3 = approximate_topological_sort(V, E, W3);
    assert(p3.first == false);
    assert(p3.second == vector<string>({"D", "A", "B", "C"}));
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

    vector<double> W2 = {4.0, 4.0, 2.0, 3.0, 1.0, 6.0, 10.0};
    auto p2 = approximate_topological_sort(V, E, W2);
    assert(p2.first == false);
    assert(p2.second == vector<string>({"C", "D", "A", "B"}));
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

    auto p = approximate_topological_sort(vars, dcs, false);
    assert(p.first == true);
    assert(p.second == vector<string>({"z", "y", "x", "u", "t"}));

    auto p2 = approximate_topological_sort(vars, dcs, true);
    assert(p2.first == true);
    assert(p2.second == vector<string>({"z", "y", "x", "u", "t"}));
}

void test_approximate_topological_sort6() {
        vector<DC<string>> dcs = {
        { {}, {"x", "y"}, 1, 10 },
        { {}, {"y", "z"}, 1, 10 },
        { {}, {"z", "u"}, 1, 10 },
        { {"x", "z"}, {"u"}, 100, 3},
        { {"y", "u"}, {"x"}, 100, 1}
    };
    vector<string> vars = { "x", "y", "z", "u" };

    auto p = approximate_topological_sort(vars, dcs, false);
    assert(p.first == false);
    assert(p.second == vector<string>({"z", "y", "x", "u"}));

    auto p2 = approximate_topological_sort(vars, dcs, true);
    assert(p2.first == false);
    assert(p2.second == vector<string>({"z", "y", "u", "x"}));
}



//==========================================================================================
// Run the test cases
//==========================================================================================

int main() {
    test_lp1();

    test_flow_bound1();
    test_flow_bound1a();
    test_flow_bound2();
    test_flow_bound3();
    test_flow_bound4();
    test_flow_bound5();
    test_flow_bound6();
    test_flow_bound7();
    test_flow_bound8();
    test_flow_bound9();
    test_flow_bound10();
    test_flow_bound_infeasible();
    test_flow_bound_JOB_Q1();

    test_flow_bound_projection1();
    test_flow_bound_projection2();
    test_flow_bound_projection3();
    test_flow_bound_projection4();

    test_flow_bound_job_join_1();

    test_approximate_topological_sort1();
    test_approximate_topological_sort2();
    test_approximate_topological_sort3();
    test_approximate_topological_sort4();
    test_approximate_topological_sort5();
    test_approximate_topological_sort6();

    return 0;
}
