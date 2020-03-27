import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from embedding_model import FA_GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F
from networkx.readwrite import json_graph
from models.graphsage.dataset import Dataset 
from collections import Counter
import os
import json


def create_small_graph(max_node_label, kara_center=2, nodes_to_remove=[]):
    karate_graph = nx.karate_club_graph()
    karate_graph.remove_nodes_from(nodes_to_remove)
    mapping = {node: i for i, node in enumerate(karate_graph.nodes())}
    center = mapping[kara_center] + max_node_label
    karate_graph = nx.relabel_nodes(karate_graph, mapping)
    edges = karate_graph.edges()
    edges = np.array(edges)
    edges += max_node_label
    return edges, center, mapping


def create_small_graph2(graph, max_node_label, center):
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    center = mapping[center] + max_node_label
    graph = nx.relabel_nodes(graph, mapping)
    edges = graph.edges()
    edges = np.array(edges)
    edges += max_node_label
    return edges, center, mapping



def read_graph(path):
    edges = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            edges.append((int(data_line[0]), int(data_line[1])))
    file.close()

    G = nx.Graph() 
    G.add_edges_from(edges)
    return G


def create_adj(edges, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        adj[edge[0], edge[1]] = 1 
        adj[edge[1], edge[0]] = 1
    return adj


def connect_two_graphs(nodes_to_concat, ori_nodes, prob_each = 0.7):
    average_deg = 3
    pseudo_edges = []
    for node in nodes_to_concat:
        if np.random.rand() < prob_each:
            to_cat = np.random.choice(ori_nodes, 3)
            pseudo_edges += [[node, ele] for ele in to_cat]
    return pseudo_edges


def evaluate(embeddings, centers, labels):
    simi = embeddings.dot(embeddings.T)
    simi_center1 = simi[centers[0]]
    arg_sort = simi_center1.argsort()[::-1]
    print("The three centers are: ")
    print(centers)
    print("{} cloest nodes to the 'center1' is: ".format(len(centers)))
    print(arg_sort[:len(centers)].tolist())
    print("The similarity values between those nodes and 'center1' is: ")
    print(simi_center1[arg_sort][:len(centers)].tolist())
    print("ACC: {:.4f}".format(jaccard_distance(arg_sort[:len(centers)].tolist(), centers)))

    print("CLUTERING RESULTs")
    print("frequency...")
    print(Counter(labels))
    labels_center = [labels[index] for index in centers]
    print("labels of center nodes: ")
    print(Counter(labels_center))
    return 1
    

def jaccard_distance(list_1, list_2):
    set1 = set(list_1)
    set2 = set(list_2)
    common = set1.intersection(set2)
    uni = set1.union(set2)
    return len(common) / len(uni)


def load_data(prefix):
    dataset = Dataset()
    dataset.load_data(prefix=prefix, normalize=True)
    return dataset


def save_graph(sub_feats, G, id_map, dataset_name, dir):
    print("Saving Graph")
    num_nodes = len(list(G.nodes))
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.81)]
    val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
    test = rand_indices[int(num_nodes * 0.9):]
    res = json_graph.node_link_data(G)    
    res['nodes'] = [
        {
            'id': node['id'],
            'val': id_map[str(node['id'])] in val,
            'test': id_map[str(node['id'])] in test
        }
        for node in res['nodes']]
                    
    res['links'] = [
        {
            'source': link['source'],
            'target': link['target']
        }
        for link in res['links']]

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")
                
    with open(dir + "/graphsage/" + dataset_name + "-G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + dataset_name + "-id_map.json", 'w') as outfile:
        json.dump(id_map, outfile)
        
    print("GraphSAGE format stored in {0}".format(dir + "/graphsage/"))
    np.save(dir + "/graphsage/" + dataset_name + '-feats.npy',sub_feats)