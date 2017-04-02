from sklearn import cluster
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def draw_communities(graph, partition, pos, algorithm_name, ax):

    communities_dict = defaultdict(list)
    for node, community in partition.items():
        communities_dict[community].append(node)

    n_communities = len(communities_dict)

    for i in range(n_communities):

        amplifier = i % 3
        multi = (i / 3) * 0.3
        red = green = blue = 0

        if amplifier == 0:
            red = 0.1 + multi
        elif amplifier == 1:
            green = 0.1 + multi
        else:
            blue = 0.1 + multi

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=communities_dict[i],
                               node_color=[0.0 + red, 0.0 + green, 0.0 + blue],
                               node_size=300,
                               alpha=0.9,
                               ax=ax)

    ax.set_title(algorithm_name)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax)


def convert_list_to_dict(list):
    return {i: item for i, item in enumerate(list)}


def convert_graph_to_edge_matrix(graph):
    # Initialize Edge Matrix
    n_nodes = len(graph)
    edge_matrix = np.zeros((n_nodes, n_nodes))

    for node in graph:
        for neighbor in graph.neighbors(node):
            edge_matrix[node][neighbor] = 1
        # edge_matrix[node][node] = 1

    return edge_matrix


if __name__ == "__main__":
    n_clusters = 2
    results = {}

    graph = nx.karate_club_graph()
    groundTruth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    pos = nx.spring_layout(graph)
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    draw_communities(graph, convert_list_to_dict(groundTruth), pos, "True clusters", ax1)

    edge_matrix = convert_graph_to_edge_matrix(graph)
    # -----------------------------------------

    # Spectral Clustering
    spectral = cluster.SpectralClustering(n_clusters=n_clusters, affinity="precomputed", n_init=200)
    spectral.fit(edge_matrix)
    results["Spectral"] = list(spectral.labels_)
    # -----------------------------------------

    # Agglomerative Clustering
    agglomerative = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    agglomerative.fit(edge_matrix)
    results["Agglomerative"] = list(agglomerative.labels_)
    # -----------------------------------------

    # K-means Clustering
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=200)
    kmeans.fit(edge_matrix)
    results["K-means"] = list(kmeans.labels_)
    # -----------------------------------------

    # Affinity Propagation Clustering
    affinity = cluster.affinity_propagation(S=edge_matrix, max_iter=200, damping=0.6)
    results["Affinity Propagation"] = list(affinity[1])
    # -----------------------------------------

    # Plot results
    for (algorithm_name, partition_list), ax in zip(results.items(), [ax3, ax4, ax5, ax6]):
        draw_communities(graph, convert_list_to_dict(partition_list), pos, algorithm_name, ax)
    plt.show()
