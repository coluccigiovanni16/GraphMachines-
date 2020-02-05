# -*- coding: utf-8 -*-
import re
from typing import List, Any

import matplotlib.pyplot as plt
import networkx as nx
import torch

from ..scott import parse, canonize


# parse a string from scott algorithm to a sequence of node->sons
def parse_newick(newick_string):
    current_molecule_token = re.findall(r"([^:;,()\s]*)(?:\s*:\s*([\d.]+)\s*)?([,);])|(\S)", newick_string + ";")

    def iter_node(next_id=0, parent_id=-1):  # one node
        current_id = next_id;
        children = []
        name, length, delimiter, character = current_molecule_token.pop(0)
        if character == "(":
            while character in "(,":
                node, character, next_id = iter_node(next_id + 1, current_id)
                children.append(node)
            name, length, delimiter, character = current_molecule_token.pop(0)
        return {"id": current_id, "name": name, "length": float(length) if length else None,
                "parent_id": parent_id, "children": children}, delimiter, next_id

    return iter_node()[0]


# have in input the path of the molecule in an sdf mode,and give in output a DAG graph
def molecule_to_networkxGraph(filename):
    compounds = parse.from_sdf(
        file_path=filename, ignore_hydrogens=True)[0]
    simple_cgraph = str(canonize.to_cgraph(compounds))
    gr = [parse_newick(simple_cgraph)]
    mol = nx.DiGraph()
    node_dict = {}
    # iter through the node
    for g in gr:
        node_dict[g["id"]] = g["name"]
        mol.add_node(g["id"])
        mol.nodes[g["id"]]["atom"] = g["name"]
        if len(g["children"]) > 0:
            for child in g["children"]:
                if child["name"] != "":
                    node_dict[child["id"]] = child["name"]
                    mol.add_edge(child["id"], child["parent_id"], weight=child["length"])
                    mol.nodes[child["id"]]["atom"] = child["name"]
                    gr.append(child)

    return mol


# load true value using basepath and testset filename
def load_true_value(base, filename_path):
    y = {}
    dataset_name = []
    filename_path = base + filename_path
    file_handle = open(filename_path, 'r')
    while True:
        # read a single line
        line = file_handle.readline()
        if not line:
            break
        value = line.split(' ')
        y[value[0]] = torch.FloatTensor([float(value[1])])
        dataset_name.append(base + value[0])
    # close the pointer to that file
    file_handle.close()
    return dataset_name, y


# iter the dataset list and return a dict with name of molecule file as key and the networkx DiGraph as value
def dict_of_file_name_list(base, dataset_name):
    graphs_dict_list = {}
    # construct graphs from every single .ct file
    for filename in dataset_name:
        graphs_dict_list[filename.replace(base, '')] = molecule_to_networkxGraph(filename.replace(".ct", ".sdf"))
    return graphs_dict_list


# return the sum of bond's weight of the edge in/out of the node in input
def get_weight_sum(graph, node):
    bond = 0
    for (u, v, d) in graph.in_edges(node, data=True):
        bond = bond + d["weight"]

    for (u, v, d) in graph.out_edges(node, data=True):
        bond = bond + d["weight"]
    return bond


# starting from the networkx graph it construct 4 dict, the name of molecule file as key and for a value:
# final_graphs_dict -> digraph ,
# sor_ordered_list  -> list of the son ordered in lexycographic manner
# depth_nodes -> list of nodes for the several depth
# centers -> the center of the DAG
# also we add the attributes at every node, used after to construct the features vector
def dag_creator(graphs_dict_list):
    final_graphs_dict = {}
    sor_ordered_list = {}
    centers = {}
    depth_nodes = {}
    atoms = {'C': [1, 0, 0, 0], 'S': [0, 1, 0, 0], 'O': [0, 0, 1, 0], 'N': [0, 0, 0, 1]}
    bond_value = {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3: [0, 1, 1], 4: [1, 0, 0], 5: [1, 0, 1]}
    for g in graphs_dict_list:
        for node in graphs_dict_list[g]:
            atom = graphs_dict_list[g].nodes[node]["atom"]
            # node attributes
            if atom in atoms:
                attrs = {node: {'attrA1': float(atoms[atom][0]), 'attrA2': float(atoms[atom][1]),
                                'attrA3': float(atoms[atom][2]), 'attrA4': float(atoms[atom][3])}}
            else:
                # if it is a node added using the scott algo(with no label) we givi it the same attributes
                attrs = {node: {'attrA1': 1, 'attrA2': 1, 'attrA3': 1, 'attrA4': 1}}
            nx.set_node_attributes(graphs_dict_list[g], attrs)
            bond = get_weight_sum(graphs_dict_list[g], node)
            # bond attributes
            attrs = {
                node: {'attrB1': bond_value[bond][0], 'attrB2': bond_value[bond][1], 'attrB3': bond_value[bond][2]}}
            nx.set_node_attributes(graphs_dict_list[g], attrs)
        final_graphs_dict[g], sor_ordered_list[g], depth_nodes[g], centers[g] = to_dag(graphs_dict_list[g])
    return final_graphs_dict, sor_ordered_list, depth_nodes, centers


def to_dag(G, plot=False):
    '''  docstring:
    converts an acyclic graph to a DAG
    with branches yet directed towards the center.
    Input: indirect graph to be converted
    Output: DAG
    '''
    if plot:
        nx.draw(G, with_labels=True, with_attributes=True)
        plt.show()
    center_dict = {}
    graph_ordered_node_trav = {}
    center = [n for n, d in G.out_degree() if d == 0][0]
    G = G.reverse()
    depth_list = nx.shortest_path_length(G, center)
    G = G.reverse()
    for n in nx.lexicographical_topological_sort(G):
        if G.in_degree(n) > 0:
            son_list_ordered = []
            for p in sorted(list(G.predecessors(n)), key=lambda x: G.nodes[x]['atom']):
                G.nodes[n]['atom'] = G.nodes[n]['atom'] + G.nodes[p]['atom']
                son_list_ordered.append(p)
            graph_ordered_node_trav[n] = son_list_ordered
        else:
            graph_ordered_node_trav[n] = []
    center_dict[G.nodes[center]['atom']] = G, graph_ordered_node_trav, depth_list, center
    return center_dict[min(center_dict.keys())]


# calculate the value D(as written on the GM paper)
def get_d_value(Graph):
    label_size_x = 7
    G = list(Graph.values())
    max_m = -1
    for g in G:
        for n in g.nodes():
            max_m = (max(max_m, g.in_degree(n)))
            # Size of the node array
        d = max_m + 1 + label_size_x
    return d, max_m


# create the features vector for every nodes of the graph in input,using
# the bias ,M and D(GM algorithm)
def create_graph_tensor(graphs, bias, max_m, d_value):
    graph_tensor = {}
    for g in graphs.keys():
        G = graphs[g]
        nodes = {}
        for n in nx.lexicographical_topological_sort(G):
            x = torch.zeros(d_value, dtype=torch.float)
            x[0] = bias
            X = [G.nodes[n].pop('attrA1'), G.nodes[n].pop('attrA2'), G.nodes[n].pop('attrA3'), G.nodes[n].pop('attrA4'),
                 G.nodes[n].pop('attrB1'),
                 G.nodes[n].pop('attrB2'), G.nodes[n].pop('attrB3')]
            if len(X) > 0:
                i = max_m + 1
                for att in X:
                    x[i] = att
                    i = i + 1
            nodes[n] = x.view(d_value, 1)
        graph_tensor[g] = nodes
    return graph_tensor


# key algorithm of my GM structure,it create the matrix used to re-compute the features vector
# for the node of the current depth,from the leaf depth to the root depth going from leaf to root
def dataset_loader(depth_nodes, center_node, sor_ordered_list, graph_tensor, label, d_value, device):
    deepthdict_batch_label = torch.zeros(0)
    deepthdict_batch_tensor = {}
    deepthdict_batch_parent_list_sons = {}
    dummy_dict = {}

    for molecule in center_node:
        deepthdict_batch_label = torch.cat([deepthdict_batch_label, label[molecule]])
        queue: List[Any] = [center_node[molecule]]
        while len(queue) > 0:
            current = queue.pop(0)
            depth = depth_nodes[molecule][current]
            if depth in deepthdict_batch_tensor.keys():
                deepthdict_batch_tensor[depth] = torch.cat(
                    [deepthdict_batch_tensor[depth], graph_tensor[molecule][current]], dim=0)
                deepthdict_batch_parent_list_sons[depth].append(len(sor_ordered_list[molecule][current]))
            else:
                deepthdict_batch_tensor[depth] = graph_tensor[molecule][current]
                deepthdict_batch_parent_list_sons[depth] = [len(sor_ordered_list[molecule][current])]
            queue.extend(sor_ordered_list[molecule][current])
    for depth in reversed(range(1, len(
            deepthdict_batch_parent_list_sons))):  # non considero il primo livello in quanto sicuramente non ha figli
        node_of_next_level = len(deepthdict_batch_parent_list_sons[depth - 1])
        node_of_level = len(deepthdict_batch_parent_list_sons[depth])
        dummy = torch.zeros(d_value * node_of_next_level, node_of_level)
        oriz = 0
        vert = 1
        for node in deepthdict_batch_parent_list_sons[depth - 1]:
            vert_temp = vert
            for son in range(node):
                dummy[vert_temp][oriz] = 1
                oriz += 1
                vert_temp += 1
            vert += d_value
        dummy_dict[depth - 1] = dummy.to_sparse().to(device)

    for depth in deepthdict_batch_tensor:
        deepthdict_batch_tensor[depth] = deepthdict_batch_tensor[depth].to(device)

    deepthdict_batch_label = deepthdict_batch_label.view(-1, 1).to(device)
    return dummy_dict, deepthdict_batch_tensor, deepthdict_batch_label
