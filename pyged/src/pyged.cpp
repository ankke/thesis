#define GUROBI
#include "src/env/ged_env.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>

using NodeLabel = std::tuple< double, double >;
using EdgeLabel = double; 


class GEDBase: public ged::EditCosts<NodeLabel, EdgeLabel>
{
public:
	double node_ins_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_del_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_rel_cost_fun(const NodeLabel& node_label_1, const NodeLabel& node_label_2) const
	{
		return 0;
	}
	
	double edge_ins_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_del_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_rel_cost_fun(const EdgeLabel& edge_label_1, const EdgeLabel& edge_label_2) const
	{
		return 0;
	}
};

class GEDNodeMatch: public ged::EditCosts<NodeLabel, EdgeLabel>
{
public:
	double node_ins_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_del_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_rel_cost_fun(const NodeLabel& node_label_1, const NodeLabel& node_label_2) const
	{
		double x1 = std::get<0>(node_label_1);
		double x2 = std::get<1>(node_label_1);
		double y1 = std::get<0>(node_label_2);
		double y2 = std::get<1>(node_label_2);

		double norm = std::sqrt(std::pow((x2 - x1), 2) + std::pow((y2 - y1), 2));

		return (norm >= 0.25) ? norm : 0;
	}
	
	double edge_ins_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_del_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_rel_cost_fun(const EdgeLabel& edge_label_1, const EdgeLabel& edge_label_2) const
	{
		return 0;
	}
};

class GEDEdgeMatch: public ged::EditCosts<NodeLabel, EdgeLabel>
{
public:
	double node_ins_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_del_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_rel_cost_fun(const NodeLabel& node_label_1, const NodeLabel& node_label_2) const
	{
		double x1 = std::get<0>(node_label_1);
		double x2 = std::get<1>(node_label_1);
		double y1 = std::get<0>(node_label_2);
		double y2 = std::get<1>(node_label_2);

		double norm = std::sqrt(std::pow((x2 - x1), 2) + std::pow((y2 - y1), 2));

		return (norm >= 0.25) ? norm : 0;
	}
	
	double edge_ins_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_del_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_rel_cost_fun(const EdgeLabel& edge_label_1, const EdgeLabel& edge_label_2) const
	{
		double len_diff = abs(edge_label_1 - edge_label_2);

		return (len_diff >= 0.1) ? len_diff : 0;
	}
};


ged::Options::GEDMethod method_name_to_option(std::string name)
{
	if (name == "branch") {
		return ged::Options::GEDMethod::BRANCH;
	} else if (name == "f2") {
		return ged::Options::GEDMethod::F2;
	} else if (name == "ipfp") {
		return ged::Options::GEDMethod::IPFP;
	} else {
		throw std::invalid_argument("unknown method");
	}
}

ged::EditCosts<NodeLabel, EdgeLabel> * cost_name_to_edit_cost(std::string name)
{
	if (name == "base") {
		return new GEDBase();
	} else if (name == "node") {
		return new GEDNodeMatch();
	} else if (name == "edge") {
		return new GEDEdgeMatch();
	} else {
		throw std::invalid_argument("unknown cost");
	}
}

using Graph = std::pair< std::vector< NodeLabel >, std::vector< std::tuple< int, int, EdgeLabel > > >;

std::tuple< double, double >
ged_dist(const Graph& g, const Graph& h, std::vector< std::string > method_name, std::vector< std::string > method_args, std::string cost_name)
{
	ged::GEDEnv< int, NodeLabel, EdgeLabel > env;
	
	auto gi = env.add_graph();
	const auto& g_x = g.first;
	const auto& g_edge_index = g.second;
	for (int i = 0; i < (int)g_x.size(); ++i) {
		env.add_node(gi, i, g_x[i]);
	}
	for (const auto& p: g_edge_index) {
		env.add_edge(gi, std::get<0>(p), std::get<1>(p), std::get<2>(p));
	}
	
	auto hi = env.add_graph();
	const auto& h_x = h.first;
	const auto& h_edge_index = h.second;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		env.add_node(hi, i, h_x[i]);
	}
	for (const auto& p: h_edge_index) {
		env.add_edge(hi, std::get<0>(p), std::get<1>(p), std::get<2>(p));
	}

	env.set_edit_costs(cost_name_to_edit_cost(cost_name));

	env.init();
	double lb, ub;
	if (method_name.size() == 1) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	} else if (method_name.size() == 2) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		env.set_method(method_name_to_option(method_name[1]), method_args[1]);
		env.init_method();
		env.run_method(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	}
	
	return std::make_tuple(lb, ub);
}

PYBIND11_MODULE(pyged, m) {
	m.def("ged_dist", &ged_dist);
}
