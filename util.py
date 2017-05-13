from sklearn import cluster
import networkx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy
import sklearn.metrics as sm
import re

from KDeterminant import KDeterminant


def get_graph_from_gml(file_name, label='label'):
    correct_gml = re.sub(r'\s+\[', ' [', open(file_name).read())
    return networkx.parse_gml(correct_gml, label)


def get_clusters_dict(partition_dict):
    clusters_dict = defaultdict(list)
    for node, cluster in partition_dict.items():
        clusters_dict[cluster].append(node)
    return clusters_dict


def draw_clusters(graph, labels, pos, algorithm_name, ax):
    clusters_dict = get_clusters_dict(convert_list_to_dict(labels))
    n_clusters = len(clusters_dict)

    for i in range(n_clusters):
        amplifier = i % 3
        red = green = blue = 0
        color = (n_clusters - i) / n_clusters
        if amplifier == 0:
            red = color
        elif amplifier == 1:
            green = color
        else:
            blue = color
        networkx.draw_networkx_nodes(G=graph, pos=pos, nodelist=clusters_dict[i], node_size=10,
                                     node_color=[red, green, blue], alpha=1, ax=ax)

    ax.set_title(algorithm_name)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    networkx.draw_networkx_edges(G=graph, pos=pos, alpha=0.2, ax=ax)


def convert_list_to_dict(list):
    return {i: item for i, item in enumerate(list)}


def AP(matrix, affinity):
    return list(cluster.AffinityPropagation(affinity=affinity).fit(matrix).labels_)


def Agglomerative(matrix, k):
    return list(cluster.AgglomerativeClustering(n_clusters=k).fit_predict(matrix))


def K_means(matrix, k):
    # k = KDeterminant().get_best_k(edge_matrix)
    return list(cluster.KMeans(n_clusters=k).fit_predict(matrix))


def Spectral(matrix, k):
    return list(cluster.SpectralClustering(n_clusters=k, affinity='precomputed').fit_predict(matrix))


def get_clustering_results(graph, k):
    adj_matrix = networkx.adjacency_matrix(graph).toarray()
    dist_matrix = networkx.floyd_warshall_numpy(graph)
    # k = KDeterminant().get_best_k(matrix)
    return {
        # "AP dist euclidean": AP(dist_matrix, 'euclidean'),
        # "AP dist precomputed": AP(dist_matrix, 'precomputed'),
        # "AP adj euclidean": AP(adj_matrix, 'euclidean'),
        # "AP adj precomputed": AP(adj_matrix, 'precomputed')
        "Agglomerative": Agglomerative(dist_matrix, k),
        "K-means": K_means(dist_matrix, k),
        "Spectral": Spectral(adj_matrix, k)
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


def plot_results(graph, results, ground_truth_info=None, filename=None):
    pos = networkx.spring_layout(graph)
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    if ground_truth_info is not None:
        draw_clusters(graph, ground_truth_info[1], pos, ground_truth_info[0], ax1)
        plot_metrics_histogram(results, ground_truth_info[1], ax2)
    for (algorithm_name, labels), ax in zip(results.items(), [ax3, ax4, ax5, ax6]):
        draw_clusters(graph, labels, pos, algorithm_name, ax)
    if filename is not None:
        f.savefig(filename + '.png')
    else:
        plt.show()


def get_test_info(filename, label, value):
    graph = get_graph_from_gml(filename, label)
    node_values = networkx.get_node_attributes(graph, value).values()
    indicators = list(set(node_values))
    ground_truth = [indicators.index(value) for value in node_values]
    return graph, ground_truth


def get_graph_info(graph):
    nodes = networkx.number_of_nodes(graph)
    edges = networkx.number_of_edges(graph)
    radius = networkx.radius(graph)
    diameter = networkx.diameter(graph)
    density = networkx.density(graph)
    average_clustering = networkx.average_clustering(graph)
    average_degree = sum(graph.degree().values()) / nodes
    return nodes, edges, radius, diameter, density, average_clustering, average_degree
