# -*- coding: utf-8 -*-
import re
from typing import List, Any

import matplotlib.pyplot as plt
import networkx as nx
import torch

from ..scott import parse, canonize


def parse_newick(newick_string):
    current_molecule_token = re.findall(r"([^:;,()\s]*)(?:\s*:\s*([\d.]+)\s*)?([,);])|(\S)", newick_string + ";")

    def recurse(next_id=0, parent_id=-1):  # one node
        current_id = next_id;
        children = []
        name, length, delimiter, character = current_molecule_token.pop(0)
        if character == "(":
            while character in "(,":
                node, character, next_id = recurse(next_id + 1, current_id)
                children.append(node)
            name, length, delimiter, character = current_molecule_token.pop(0)
        return {"id": current_id, "name": name, "length": float(length) if length else None,
                "parent_id": parent_id, "children": children}, delimiter, next_id

    return recurse()[0]


def molToNetworkx(filename):
    compounds = parse.from_sdf(
        file_path=filename, ignore_hydrogens=True)[0]
    simple_cgraph = str(canonize.to_cgraph(compounds))
    gr = [parse_newick(simple_cgraph)]
    mol = nx.DiGraph()
    node_dict = {}
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


# %%

def load_true_value(base, filename_path):
    Y = {}
    dataset_name = []
    filename_path = base + filename_path
    file_handle = open(filename_path, 'r')
    while True:
        # read a single line
        line = file_handle.readline()
        if not line:
            break
        value = line.split(' ')
        Y[value[0]] = torch.FloatTensor([float(value[1])])
        dataset_name.append(base + value[0])
    # close the pointer to that file
    file_handle.close()
    return dataset_name, Y


def dictOfFNameList(base, dataset_name):
    graphs_dict_list = {}
    # construct graphs from every single .ct file
    for filename in dataset_name:
        graphs_dict_list[filename.replace(base, '')] = molToNetworkx(filename.replace(".ct", ".sdf"))
    return graphs_dict_list


def get_weight_sum(graph, node):
    bond = 0
    #     print("in",graph.in_edges(node,data=True))
    #     print("in",graph.out_edges(node,data=True))
    for (u, v, d) in graph.in_edges(node, data=True):
        bond = bond + d["weight"]

    for (u, v, d) in graph.out_edges(node, data=True):
        bond = bond + d["weight"]
    return bond


def dag_dict(graphs_dict_list):
    final_graphs_dict = {}
    sor_ordered_list = {}
    centers = {}
    depth_nodes = {}
    atoms = {'C': [1, 0, 0, 0], 'S': [0, 1, 0, 0], 'O': [0, 0, 1, 0], 'N': [0, 0, 0, 1]}
    bond_value = {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3: [0, 1, 1], 4: [1, 0, 0], 5: [1, 0, 1]}
    for g in graphs_dict_list:
        for node in graphs_dict_list[g]:
            atom = graphs_dict_list[g].nodes[node]["atom"]
            if atom in atoms:
                attrs = {node: {'attrA1': float(atoms[atom][0]), 'attrA2': float(atoms[atom][1]),
                                'attrA3': float(atoms[atom][2]), 'attrA4': float(atoms[atom][3])}}
            else:
                attrs = {node: {'attrA1': 1, 'attrA2': 1, 'attrA3': 1, 'attrA4': 1}}
            nx.set_node_attributes(graphs_dict_list[g], attrs)
            bond = get_weight_sum(graphs_dict_list[g], node)
            attrs = {
                node: {'attrB1': bond_value[bond][0], 'attrB2': bond_value[bond][1], 'attrB3': bond_value[bond][2]}}
            nx.set_node_attributes(graphs_dict_list[g], attrs)
        final_graphs_dict[g], sor_ordered_list[g], depth_nodes[g], centers[g] = to_dag(graphs_dict_list[g])
    return final_graphs_dict, sor_ordered_list, depth_nodes, centers


# %%

def to_dag(G, plot=False):
    '''  docstring:
    converte un grafo aciclico in un grafo dag
    con i rami già diretti verso il centro.
    Input : grafo indiretto da covertire
    Output : grafo dag con grafico
    '''
    if plot:
        nx.draw(G, with_labels=True, with_attributes=True)
        plt.show()
    center_dict = {}
    #     src = nx.center(G)
    #     for center in src:
    graph_ordered_node_trav = {}
    #         T = nx.dfs_tree(G, source=center)
    #         G.remove_edges_from(list(G.edges()))
    #         G = G.to_directed()
    #         G.add_edges_from(list(T.edges()))
    center = [n for n, d in G.out_degree() if d == 0][0]
    #     print(center)
    G = G.reverse()
    depth_list = nx.shortest_path_length(G, center)
    #     print(depth_list)
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


def getDvalue(Graph):
    labelSizeX = 7
    G = list(Graph.values())
    maxM = -1
    for g in G:
        for n in g.nodes():
            maxM = (max(maxM, g.in_degree(n)))
            # Size of the node array
        D = maxM + 1 + labelSizeX
    return D, maxM


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


def dataset_loader(depth_nodes, center_node, sor_ordered_list, graph_tensor, label, d_value, device):
    deepthdict_batch_label = torch.zeros(0)
    deepthdict_batch_tensor = {}
    deepthdict_batch_parent_list_sons = {}
    # dummy_dict = {}

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
    dummy_dict = {}
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
            vert_temp = +1
            vert += d_value
        dummy_dict[depth - 1] = dummy.to_sparse().to(device)

    for depth in deepthdict_batch_tensor:
        deepthdict_batch_tensor[depth] = deepthdict_batch_tensor[depth].to(device)

    deepthdict_batch_label = deepthdict_batch_label.view(-1, 1).to(device)
    return dummy_dict, deepthdict_batch_tensor, deepthdict_batch_label
