import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_nodes_from(range(3))
G.add_edge(0, 1, weight = 1 )
#G.add_edge(0, 2, weight = 1 )
#G.add_edge(1, 2, weight = 3 )


nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

def visualize(edges):
    G_1 = nx.Graph()
    G_1.add_edges_from(edges)
    #n_nodes = len(G_1.nodes)
    #pos = {i : ( self.positions[G_1.nodes[i]][0] , self.positions[G_1.nodes[i]][0] ) for i in range(n_nodes)}
    #nx.draw(G_1, pos, edge_labels = True, with_labels=True, font_weight='bold')
    nx.draw(G_1, edge_labels = True, with_labels=True, font_weight='bold')
    plt.show(block = False)


if __name__ == '__main__': pass