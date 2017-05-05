from sklearn import cluster
import networkx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy
import sklearn.metrics as sm
import re

from KDeterminant import KDeterminant


def get_graph_from_gml_file(file_name, label='label'):
    correct_gml = re.sub(r'\s+\[', ' [', open(file_name).read())
    return networkx.parse_gml(correct_gml, label)


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

        networkx.draw_networkx_nodes(G=graph,
                                     pos=pos,
                                     nodelist=communities_dict[i],
                                     node_size=50,
                                     node_color=[red, green, blue],
                                     alpha=1,
                                     ax=ax)

    ax.set_title(algorithm_name)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    networkx.draw_networkx_edges(G=graph, pos=pos, alpha=0.2, ax=ax)


def convert_list_to_dict(list):
    return {i: item for i, item in enumerate(list)}


def convert_graph_to_edge_matrix(graph):
    n_nodes = len(graph)
    edge_matrix = numpy.zeros((n_nodes, n_nodes))

    for node in graph:
        for neighbor in graph.neighbors(node):
            edge_matrix[node][neighbor] = 1
        edge_matrix[node][node] = 1

    return edge_matrix


def get_clustering_results(graph):
    edge_matrix = convert_graph_to_edge_matrix(graph)
    k = KDeterminant().get_best_k(edge_matrix)
    print(k)

    return {
        "Affinity Propagation": list(cluster.AffinityPropagation().fit(edge_matrix).labels_),
        "Agglomerative": list(cluster.AgglomerativeClustering(n_clusters=k).fit(edge_matrix).labels_),
        "K-means": list(cluster.KMeans(n_clusters=k).fit(edge_matrix).labels_),
        "Spectral": list(cluster.SpectralClustering(n_clusters=k).fit(edge_matrix).labels_)
    }


def get_clustering_quality(ground_truth, labels):
    metrics = [sm.normalized_mutual_info_score, sm.adjusted_rand_score, sm.v_measure_score, sm.fowlkes_mallows_score]

    return numpy.array([metric(ground_truth, labels) for metric in metrics]).mean()


def plot_metrics_histogram(results, ground_truth, ax):
    y = [get_clustering_quality(ground_truth, labels) for labels in results.values()]

    ind = numpy.arange(len(y))
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


def plot_results(graph, results, ground_truth=None):
    pos = networkx.spring_layout(graph)
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    if ground_truth is not None:
        draw_communities(graph, ground_truth, pos, "True clusters", ax1)
        plot_metrics_histogram(results, ground_truth, ax2)
    for (algorithm_name, labels), ax in zip(results.items(), [ax3, ax4, ax5, ax6]):
        draw_communities(graph, labels, pos, algorithm_name, ax)
    plt.show()


def get_test_info(filename, label, value):
    graph = get_graph_from_gml_file(filename, label)
    node_values = networkx.get_node_attributes(graph, value).values()
    indicators = list(set(node_values))
    ground_truth = [indicators.index(value) for value in node_values]
    return graph, ground_truth


def print_info(graph):
    nodes = networkx.number_of_nodes(graph)
    print("Nodes: ", nodes)
    print("Edges: ", networkx.number_of_edges(graph))
    print("Radius: ", networkx.radius(graph))
    print("Diameter: ", networkx.diameter(graph))
    print("Density: ", networkx.density(graph))

    print("Average clustering coefficient: ", networkx.average_clustering(graph))
    print("Average degree: ", sum(graph.degree().values()) / nodes)


def get_graph_info(graph):
    nodes = networkx.number_of_nodes(graph)
    edges = networkx.number_of_edges(graph)
    radius = networkx.radius(graph)
    diameter = networkx.diameter(graph)
    density = networkx.density(graph)
    average_clustering = networkx.average_clustering(graph)
    average_degree = sum(graph.degree().values()) / nodes
    return nodes, edges, radius, diameter, density, average_clustering, average_degree
