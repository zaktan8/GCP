import networkx
import util
import csv
import pandas

if __name__ == "__main__":

    graphs_with_ground_truth = {}

    m1, m2 = 20, 8
    graphs_with_ground_truth['barbell_{}_{}'.format(m1, m2)] = (networkx.barbell_graph(m1, m2),
                                                                [0] * m1 + [1] * m2 + [2] * m1)
    graphs_with_ground_truth['adjnoun'] = util.get_test_info("graphs/adjnoun.gml", "id", "value")
    graphs_with_ground_truth['football'] = util.get_test_info("graphs/football.gml", "id", "value")
    graphs_with_ground_truth['polbooks'] = util.get_test_info("graphs/polbooks.gml", "id", "value")

    graph = networkx.karate_club_graph()
    node_values = [graph.node[i]['club'] for i in graph.nodes()]
    indicators = list(set(node_values))
    ground_truth = [indicators.index(value) for value in node_values]
    graphs_with_ground_truth['karate_club'] = (graph, ground_truth)

    with open('graphs_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['graph_name', 'nodes', 'edges', 'radius',
                   'diameter', 'density', 'avg_clustering_coeff',
                   'avg_degree', 'best_algorithm', 'quality_score']
        writer.writerow(headers)
        for graph_name, (graph, ground_truth) in graphs_with_ground_truth.items():
            clustering_results = util.get_clustering_results(graph)
            scores = {algo_name: util.get_clustering_quality(ground_truth, labels)
                      for algo_name, labels in clustering_results.items()}
            best_algo = max(clustering_results, key=scores.get)
            writer.writerow([graph_name, *util.get_graph_info(graph), best_algo, scores[best_algo]])
            # util.plot_results(graph, clustering_results, ground_truth)
    pass

    data = pandas.read_csv("graphs_info.csv")
    pandas.set_option('precision', 4)
    print(data)
    # print(data.sort_values(by=data.columns[6], ascending=False))
