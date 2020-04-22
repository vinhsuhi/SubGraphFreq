import os 
import networkx as nx


def read_source(path):
    # G = nx.Graph
    edges = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if 'e' in line:
                data_line = line.split()
                edge_0, edge_1 = int(data_line[1]), int(data_line[2]), int(data_line[3])
                edges.append([edge_0, edge_1])
            elif 'v' in line:
                
    file.close()

    return edges


def save_edge_list(edges, to_save):
    with open(to_save, 'w', encoding='utf-8') as file:
        for edge in edges:
            file.write("{}\t{}\t{}\n".format(*edge))
    file.close()
    



if __name__ == "__main__":
    source_path = 'GraMi/Datasets/mico.lg'
    target_path = 'data/mico.edges'
    save_edge_list(read_source(source_path), target_path)
    print("DONE!")