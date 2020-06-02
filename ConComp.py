from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter
from functools import reduce
import networkx as nx

"""
Transforms the list of edges to a matrix
"""
def graphMaker(edges, n):
    graph = [[ 0 for j in range(n + 1)] for i in range(n + 1)]
    for edge in edges:
        i, j = edge
        graph[i][j] = 1
    return graph


"""
Given a list of undirected edges with nodes indexes and the number n of
nodes of the graph, the function returns a list of lists representing the
subsets that should generate the cut.
"""
def getSubsets(edges, n):
    graph = csr_matrix(graphMaker(edges, n))
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    if labels[0] != 0:
        print('Format Input Error!')
        return []
    subsets = []
    for key in Counter(labels):
        if key >= 1 and Counter(labels)[key] >= 2:
            subsets.append([i for i in range(len(labels)) if labels[i] == key])
    return subsets


"""
Given a list of undirected edges with nodes indexes and the number n of
nodes of the graph, the function returns a list of lists representing the
subsets that should generate the cut.
"""
def MFComponents(edges, n):
    # Graph is created.
    G = nx.DiGraph()
    for edge in edges:
        i, j = edge
        G.add_node(i)
        G.add_node(j)
        G.add_edge(i, j, capacity = 1)
        G.add_edge(j, i, capacity = 1)

    CC = [ ]
    while True:
        if len(CC) == 0:
            missing = set(G.nodes)
        else:
            missing = list(set(G.nodes).difference(set( reduce(lambda x, y : x + y, CC) )))

        if len(missing) == 0:
            break
        newNode = missing.pop()
        newCC = [newNode]
        for i in missing:
            flow_value, flow_dict = nx.maximum_flow(G, newNode, i)
            if flow_value >= 1:
                newCC.append(i)
        CC.append(newCC)
    
    print(CC)

    return CC



if __name__ == '__main__':
    EX1 = [(0, 3), (1, 2), (1, 4), (2, 4)]
    EX2 = [(0, 1), (0, 2), (1, 4), (2, 4)]
    EX3 = [(0, 2)]
    EX4 = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    EX5 = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (6, 7), (7, 8), (6, 8)]
    EX6 = [(0, 1), (2, 3)]
    EX7 = [(0, 4), (1, 3), (3, 6), (1, 6)]
    EX8 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]


    #print(getSubsets(EX1, 10))
    #print(getSubsets(EX2, 10))
    #print(getSubsets(EX3, 10))
    #print(getSubsets(EX4, 10))
    #print(getSubsets(EX5, 10))
    #print(getSubsets(EX6, 10))
    MFComponents(EX5, 8)
