from sklearn import cluster
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
import re

from KDeterminant import KDeterminant


def get_graph_from_gml_file(file_name, label='label'):
    correct_gml = re.sub(r'\s+\[', ' [', open(file_name).read())
    return nx.parse_gml(correct_gml, label)


def get_communities_dict(partition_dict):
    communities_dict = defaultdict(list)
    for node, community in partition_dict.items():
        communities_dict[community].append(node)
    return communities_dict


def draw_communities(graph, labels, pos, algorithm_name, ax):
    communities_dict = get_communities_dict(convert_list_to_dict(labels))
    n_communities = len(communities_dict)

    for i in range(n_communities):

        amplifier = i % 3
        red = green = blue = 0

        color = (n_communities - i) / n_communities

        if amplifier == 0:
            red = color
        elif amplifier == 1:
            green = color
        else:
            blue = color

        nx.draw_networkx_nodes(G=graph,
                               pos=pos,
                               nodelist=communities_dict[i],
                               node_size=50,
                               node_color=[red, green, blue],
                               alpha=1,
                               ax=ax)

    ax.set_title(algorithm_name)
    nx.draw_networkx_edges(G=graph, pos=pos, alpha=0.2, ax=ax)
    # labels = {i: label for i, label in enumerate(graph.nodes())}
    # nx.draw_networkx_labels(graph, pos, labels, font_size=16)


def convert_list_to_dict(list):
    return {i: item for i, item in enumerate(list)}


def convert_graph_to_edge_matrix(graph):
    n_nodes = len(graph)
    edge_matrix = np.zeros((n_nodes, n_nodes))

    for node in graph:
        for neighbor in graph.neighbors(node):
            edge_matrix[node][neighbor] = 1
        edge_matrix[node][node] = 1

    return edge_matrix


def get_clustering_quality(ground_truth, labels):
    metrics = [sm.normalized_mutual_info_score, sm.adjusted_rand_score, sm.v_measure_score]
    return np.array([metric(ground_truth, labels) for metric in metrics]).mean()


def plot_metrics_histogram(results, ground_truth, ax):
    y = [get_clustering_quality(ground_truth, labels) for labels in results.values()]

    ind = np.arange(len(y))
    width = 0.35

    ax.bar(ind, y, width, color='blue', error_kw=dict(elinewidth=2, ecolor='red'))
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel('Average Metrics Score')
    ax.set_title('Score Evaluation')

    ax.set_xticks(ind + width / 2)
    xtickNames = ax.set_xticklabels(results.keys())
    plt.setp(xtickNames, fontsize=7)

    for i, score in enumerate(y):
        ax.text(i - width / 2, score, str(round(score, 2)), color='blue', fontweight='bold')


if __name__ == "__main__":
    graph = get_graph_from_gml_file("graphs/karate.gml", "id")
    graph = nx.convert_node_labels_to_integers(graph)
    ground_truth = [0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 0, 0, 0, 0, 1, 1,
                    0, 0, 1, 0, 1, 0, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1]
    n_clusters = len(set(ground_truth))

    # graph = nx.barbell_graph(5, 6)
    # ground_truth = [0, 0, 0, 0, 0,
    #                 1, 1, 1, 1, 1, 1,
    #                 2, 2, 2, 2, 2]
    # n_clusters = len(set(ground_truth))

    # graph = get_graph_from_gml_file("graphs/adjnoun.gml", "id")
    # node_values = nx.get_node_attributes(graph, "value").values()
    # indicators = list(set(node_values))
    # ground_truth = [indicators.index(value) for value in node_values]
    # n_clusters = len(indicators)

    # graph = get_graph_from_gml_file("graphs/football.gml", "id")
    # node_values = nx.get_node_attributes(graph, "value").values()
    # indicators = list(set(node_values))
    # ground_truth = [indicators.index(value) for value in node_values]
    # n_clusters = len(indicators)

    # graph = get_graph_from_gml_file("graphs/polbooks.gml", "id")
    # node_values = nx.get_node_attributes(graph, "value").values()
    # indicators = list(set(node_values))
    # ground_truth = [indicators.index(value) for value in node_values]
    # n_clusters = len(indicators)

    edge_matrix = convert_graph_to_edge_matrix(graph)
    # -----------------------------------------

    results = {}

    # Spectral Clustering
    spectral = cluster.SpectralClustering(n_clusters=n_clusters).fit(edge_matrix)
    results["Spectral"] = list(spectral.labels_)
    # -----------------------------------------

    # Agglomerative Clustering
    agglomerative = cluster.AgglomerativeClustering().fit(edge_matrix)
    results["Agglomerative"] = list(agglomerative.labels_)
    # -----------------------------------------

    # K-means Clustering
    k = KDeterminant().get_best_k(edge_matrix)
    kmeans = cluster.KMeans(n_clusters=k).fit(edge_matrix)
    results["K-means"] = list(kmeans.labels_)
    # -----------------------------------------

    # Affinity Propagation Clustering
    affinity = cluster.AffinityPropagation().fit(edge_matrix)
    results["Affinity Propagation"] = list(affinity.labels_)

    # Plot clustering results
    pos = nx.spring_layout(graph)
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3
                                                         # , sharex='col', sharey='row'
                                                         )
    draw_communities(graph, ground_truth, pos, "True clusters", ax1)

    for (algorithm_name, labels), ax in zip(results.items(), [ax3, ax4, ax5, ax6]):
        draw_communities(graph, labels, pos, algorithm_name, ax)

    plot_metrics_histogram(results, ground_truth, ax2)

    plt.show()
