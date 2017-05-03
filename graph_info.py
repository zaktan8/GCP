from main import *


def print_info(graph):
    print("Name: ", graph.name)
    nodes = nx.number_of_nodes(graph)
    print("Nodes: ", nodes)
    print("Edges: ", nx.number_of_edges(graph))
    print("Radius: ", nx.radius(graph))
    print("Diameter: ", nx.diameter(graph))
    print("Density: ", nx.density(graph))

    print("Average clustering coefficient: ", nx.average_clustering(graph))
    print("Average degree: ", sum(graph.degree().values()) / nodes)
    print("Average degree centrality: ", sum(nx.degree_centrality(graph).values()) / nodes)


if __name__ == "__main__":
    # graph = nx.karate_club_graph()
    # graph = nx.florentine_families_graph()
    # graph = nx.davis_southern_women_graph()
    # graph = nx.barbell_graph(5, 10)
    graph = get_graph_from_gml_file('karate.gml')

    print_info(graph)
    nx.draw_networkx(graph)
    plt.show()
