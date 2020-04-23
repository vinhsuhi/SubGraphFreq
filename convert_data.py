import os 
import networkx as nx
import numpy as np

def read_source(path):
    # G = nx.Graph
    edges = []
    nodes = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            if 'v' in line:
                v0, label = int(data_line[1]), int(data_line[2])
                if label > 10:
                    label = np.random.randint(1, 11, 1)[0]
                nodes.append((v0, label))
                
            if 'e' in line:
                edge_0, edge_1, edge_label = int(data_line[1]), int(data_line[2]), int(data_line[3])
                # edges.append([edge_0, edge_1])
                if edge_label > 3:
                    edge_label = np.random.randint(0, 4, 1)[0]
                edges.append((edge_0, edge_1, edge_label))
            
    file.close()

    return nodes, edges


def save_edge_list(edges, nodes, to_save):
    with open(to_save, 'w', encoding='utf-8') as file:
        file.write('t # 1\n')
        for node in nodes:
            file.write("v {} {}\n".format(*node))
        for edge in edges:
            file.write("e {} {} {}\n".format(*edge))
    file.close()
    



if __name__ == "__main__":
    source_path = 'GraMi/Datasets/mico.lg'
    target_path = 'GraMi/Datasets/mico_10_3.lg'
    nodes, edges = read_source(source_path)
    save_edge_list(edges, nodes, target_path)
    # save_edge_list(read_source(source_path), target_path)
    print("DONE!")