# -*- coding: utf-8 -*-
import glob
import logging
import os
import random
import re
import tarfile


def parse(newick):
    tokens = re.findall(r"([^:;,()\s]*)(?:\s*:\s*([\d.]+)\s*)?([,);])|(\S)", newick + ";")

    def recurse(nextid=0, parentid=-1):  # one node
        thisid = nextid;
        children = []
        name, length, delim, ch = tokens.pop(0)
        if ch == "(":
            while ch in "(,":
                node, ch, nextid = recurse(nextid + 1, thisid)
                children.append(node)
            name, length, delim, ch = tokens.pop(0)
        return {"id": thisid, "name": name, "length": float(length) if length else None,
                "parentid": parentid, "children": children}, delim, nextid

    return recurse()[0]


def molToNetworkx(filename):
    compounds = st.parse.from_sdf(
        file_path=filename, ignore_hydrogens=True)[0]
    simple_cgraph = str(st.canonize.to_cgraph(compounds))
    gr = [parse(simple_cgraph)]
    mol = nx.DiGraph()
    nodeDict = {}
    for g in gr:
        nodeDict[g["id"]] = g["name"]
        mol.add_node(g["id"])
        mol.nodes[g["id"]]["atom"] = g["name"]
        if len(g["children"]) > 0:
            for child in g["children"]:
                if child["name"] != "":
                    nodeDict[child["id"]] = child["name"]
                    mol.add_edge(child["id"], child["parentid"], weight=child["length"])
                    mol.nodes[child["id"]]["atom"] = child["name"]
                    gr.append(child)

    return mol


# %%

def loadY(base, filenamepath):
    Y = {}
    datasetname = []
    filenamepath = base + filenamepath
    filehandle = open(filenamepath, 'r')
    while True:
        # read a single line
        line = filehandle.readline()
        if not line:
            break
        value = line.split(' ')
        Y[value[0]] = torch.FloatTensor([float(value[1])])
        datasetname.append(base + value[0])
    # close the pointer to that file
    filehandle.close()
    return datasetname, Y


# %%

# rootDir = './PAH/'
# i=0
# trainFile = "trainset_"+str(i)+".ds"
# testFile = "testset_"+str(i)+".ds"
# y=loadY(rootDir,testFile)
# print(y)

# rootDir = './Alkane/'
# i=0
# trainFile = "trainset_"+str(i)+".ds"
# testFile = "testset_"+str(i)+".ds"
# y=loadY(rootDir,testFile)
# print(y)


# %%

def dictOfFNameList(base, datasetname):
    GraphsDictList = {}
    # construct graphs from every single .ct file
    for filename in datasetname:
        GraphsDictList[filename.replace(base, '')] = molToNetworkx(filename.replace(".ct", ".sdf"))
    return GraphsDictList


def getWeightSum(graph, node):
    bond = 0
    #     print("in",graph.in_edges(node,data=True))
    #     print("in",graph.out_edges(node,data=True))
    for (u, v, d) in graph.in_edges(node, data=True):
        bond = bond + d["weight"]

    for (u, v, d) in graph.out_edges(node, data=True):
        bond = bond + d["weight"]
    return bond


# mol=molToNetworkx("C:/Users/jiovy/Documents/GitHub/scott/data/molecule/cafeine.sdf")
# for node in mol:
#     print(getWeightSum(mol,node))
# graphTrain, sorOrderedListTrain, depthNodesTrain, centerNodeTrain = DAGDict(
#         dictOfFNameList(rootDir, datasetFilenameTrain))

# %%

def DAGDict(GraphsDictList):
    finalGraphsDict = {}
    sorOrderedList = {}
    centers = {}
    depthNodes = {}
    atoms = {'C': [1, 0, 0, 0], 'S': [0, 1, 0, 0], 'O': [0, 0, 1, 0], 'N': [0, 0, 0, 1]}
    bondValue = {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3: [0, 1, 1], 4: [1, 0, 0], 5: [1, 0, 1]}
    for g in GraphsDictList:
        for node in GraphsDictList[g]:
            atom = GraphsDictList[g].nodes[node]["atom"]
            if atom in atoms:
                attrs = {node: {'attrA1': float(atoms[atom][0]), 'attrA2': float(atoms[atom][1]),
                                'attrA3': float(atoms[atom][2]), 'attrA4': float(atoms[atom][3])}}
            else:
                attrs = {node: {'attrA1': 1, 'attrA2': 1, 'attrA3': 1, 'attrA4': 1}}
            nx.set_node_attributes(GraphsDictList[g], attrs)
            bond = getWeightSum(GraphsDictList[g], node)
            attrs = {node: {'attrB1': bondValue[bond][0], 'attrB2': bondValue[bond][1], 'attrB3': bondValue[bond][2]}}
            nx.set_node_attributes(GraphsDictList[g], attrs)
        finalGraphsDict[g], sorOrderedList[g], depthNodes[g], centers[g] = toDAG(GraphsDictList[g])
    return finalGraphsDict, sorOrderedList, depthNodes, centers


# %%

def toDAG(G, plot=False):
    '''  docstring:
    converte un grafo aciclico in un grafo dag
    con i rami già diretti verso il centro.
    Input : grafo indiretto da covertire
    Output : grafo dag con grafico
    '''
    if (plot):
        nx.draw(G, with_labels=True, with_attributes=True)
        plt.show()
    centerDict = {}
    #     src = nx.center(G)
    #     for center in src:
    graphOrderedNodeTrav = {}
    #         T = nx.dfs_tree(G, source=center)
    #         G.remove_edges_from(list(G.edges()))
    #         G = G.to_directed()
    #         G.add_edges_from(list(T.edges()))
    center = [n for n, d in G.out_degree() if d == 0][0]
    #     print(center)
    G = G.reverse()
    depthList = nx.shortest_path_length(G, center)
    #     print(depthList)
    G = G.reverse()
    for n in nx.lexicographical_topological_sort(G):
        if (G.in_degree(n) > 0):
            sonListOrdered = []
            for p in sorted(list(G.predecessors(n)), key=lambda x: G.nodes[x]['atom']):
                G.nodes[n]['atom'] = G.nodes[n]['atom'] + G.nodes[p]['atom']
                sonListOrdered.append(p)
            graphOrderedNodeTrav[n] = sonListOrdered
        else:
            graphOrderedNodeTrav[n] = []
    centerDict[G.nodes[center]['atom']] = G, graphOrderedNodeTrav, depthList, center
    return centerDict[min(centerDict.keys())]


# %%

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


# %%

def createGraphTensor(Graphs, bias, maxM, DValue):
    graphTensor = {}
    for g in Graphs.keys():
        G = Graphs[g]
        nodes = {}
        for n in nx.lexicographical_topological_sort(G):
            x = torch.zeros(DValue, dtype=torch.float)
            x[0] = bias
            X = [G.nodes[n].pop('attrA1'), G.nodes[n].pop('attrA2'), G.nodes[n].pop('attrA3'), G.nodes[n].pop('attrA4'),
                 G.nodes[n].pop('attrB1'),
                 G.nodes[n].pop('attrB2'), G.nodes[n].pop('attrB3')]
            if (len(X) > 0):
                i = maxM + 1
                for att in X:
                    x[i] = att
                    i = i + 1
            nodes[n] = x.view(DValue, 1)
        graphTensor[g] = nodes
    return graphTensor


# %%

def datasetLoader(depthNodes, centerNode, sorOrderedList, graphTensor, label, DValue, device):
    deepthdictBatchLabel = torch.zeros(0)
    deepthdictBatchTensor = {}
    deepthdictBatchParentListSons = {}
    dummyDict = {}

    for molecule in centerNode:
        deepthdictBatchLabel = torch.cat([deepthdictBatchLabel, label[molecule]])
        queue = []
        queue.append(centerNode[molecule])
        while (len(queue) > 0):
            current = queue.pop(0)
            depth = depthNodes[molecule][current]
            if depth in deepthdictBatchTensor.keys():
                deepthdictBatchTensor[depth] = torch.cat([deepthdictBatchTensor[depth], graphTensor[molecule][current]],
                                                         dim=0)
                deepthdictBatchParentListSons[depth].append(len(sorOrderedList[molecule][current]))
            else:
                deepthdictBatchTensor[depth] = graphTensor[molecule][current]
                deepthdictBatchParentListSons[depth] = [len(sorOrderedList[molecule][current])]
            queue.extend(sorOrderedList[molecule][current])
    dummyDict = {}
    for depth in reversed(range(1, len(
            deepthdictBatchParentListSons))):  # non considero il primo livello in quanto sicuramente non ha figli
        nodeOfNextLevel = len(deepthdictBatchParentListSons[depth - 1])
        nodeOfLevel = len(deepthdictBatchParentListSons[depth])
        dummy = torch.zeros(DValue * nodeOfNextLevel, nodeOfLevel)
        oriz = 0
        vert = 1
        for node in deepthdictBatchParentListSons[depth - 1]:
            vertTemp = vert
            for son in range(node):
                dummy[vertTemp][oriz] = 1
                oriz += 1
            vertTemp = +1
            vert += DValue
        dummyDict[depth - 1] = dummy.to_sparse().to(device)

    for depth in deepthdictBatchTensor:
        deepthdictBatchTensor[depth] = deepthdictBatchTensor[depth].to(device)

    deepthdictBatchLabel = deepthdictBatchLabel.view(-1, 1).to(device)
    return dummyDict, deepthdictBatchTensor, deepthdictBatchLabel