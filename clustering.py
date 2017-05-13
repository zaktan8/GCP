import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx
import util
import csv
import numpy as np

if __name__ == "__main__":
    graphs_with_ground_truth = {}

    min_cluster_size = 10
    for p_out in np.arange(0.2, 0.5, 0.1):
        for p_in in np.arange(0.5, 0.9, 0.1):
            for k in range(3, 10):
                for max_cluster_size in range(min_cluster_size, 101, 10):
                    sizes = [random.randint(min_cluster_size, max_cluster_size) for i in range(k)]
                    graph = networkx.random_partition_graph(sizes, p_in, p_out)
                    ground_truth = [i for i, partition in enumerate(graph.graph['partition']) for node in partition]
                    graph_name = "rp_{:.1f}_{:.1f}_{}_{}_{}".format(p_out, p_in, k, min_cluster_size, max_cluster_size)
                    # util.plot_results(graph, util.get_clustering_results(graph, k), (graph_name, ground_truth))
                    graphs_with_ground_truth[graph_name] = (graph, ground_truth, k)

    accuracy = defaultdict(list)
    with open('graphs_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['graph_name', 'nodes', 'edges',
                   'radius', 'diameter', 'density',
                   'avg_cl_coeff', 'avg_degree',
                   'best_score', 'best_algorithm']
        writer.writerow(headers)

        for graph_name, (graph, ground_truth, k) in graphs_with_ground_truth.items():
            clustering_results = util.get_clustering_results(graph, k)
            scores = {}
            print(graph_name)
            # scores = {algo_name: util.get_clustering_quality(ground_truth, labels)
            #           for algo_name, labels in clustering_results.items()}
            for algo_name, labels in clustering_results.items():
                score = util.get_clustering_quality(ground_truth, labels)
                scores[algo_name] = score
                accuracy[algo_name].append(score)
            best_algo_name = max(scores, key=scores.get)
            best_score = scores[best_algo_name]
            try:
                if best_score > 0.7:
                    writer.writerow([graph_name, *util.get_graph_info(graph), best_score, best_algo_name])
            except networkx.NetworkXError:
                print('oops with {}'.format(graph_name))
                continue

    print('\n')
    accuracy_higher_than_0_7 = {algo_name: list(filter(lambda x: x > 0.7, scores)) for algo_name, scores in accuracy.items()}
    for k, v in accuracy_higher_than_0_7.items():
        print(k, ': ', len(v))
    print('\n')
    mean_accuracy_dict = {algo_name: np.array(scores).mean() for algo_name, scores in accuracy_higher_than_0_7.items()}
    for algo_name in sorted(mean_accuracy_dict, key=mean_accuracy_dict.get, reverse=True):
        print('{}:\t{:.3f}'.format(algo_name, mean_accuracy_dict[algo_name]))
