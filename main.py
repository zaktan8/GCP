import networkx
import util
import csv
import pandas

if __name__ == "__main__":
    # Clustering
    # m1, m2 = 20, 8
    # graph = networkx.barbell_graph(m1, m2)
    # ground_truth = [0] * m1
    # ground_truth += [1] * m2
    # ground_truth += [2] * m1

    # graph, ground_truth = util.get_test_info("graphs/adjnoun.gml", "id", "value")
    # graph, ground_truth = util.get_test_info("graphs/football.gml", "id", "value")
    graph, ground_truth = util.get_test_info("graphs/polbooks.gml", "id", "value")

    # graph = get_graph_from_gml_file("graphs/dolphins.gml", "id")
    # ground_truth = None

    # graph = networkx.karate_club_graph()
    # node_values = [graph.node[i]['club'] for i in graph.nodes()]
    # indicators = list(set(node_values))
    # ground_truth = [indicators.index(value) for value in node_values]

    results = util.get_clustering_results(graph)
    util.plot_results(graph, results, ground_truth)
    # ----------------------------

    # Classification
    # graph_filenames = ['graphs/adjnoun.gml', 'graphs/dolphins.gml',
    #                    'graphs/football.gml', 'graphs/polbooks.gml']
    #
    # graphs = [util.get_graph_from_gml_file(filename, 'id') for filename in graph_filenames]
    # graphs.append(networkx.barbell_graph(20, 8))
    # graphs.append(networkx.karate_club_graph())
    #
    # headers = ['nodes', 'edges', 'radius', 'diameter', 'density', 'avg_clustering_coeff', 'avg_degree']
    # graphs_info = [util.get_graph_info(gr) for gr in graphs]
    #
    # with open('graphs_info.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(headers)
    #     writer.writerows(graphs_info)
    #
    # data = pandas.read_csv("graphs_info.csv")
    # print(data)
    # print(data.sort_values(by=data.columns[6], ascending=False))
    # ----------------------------
