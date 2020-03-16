from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import _pickle as pickle 
import pdb
import json
import os
from networkx.readwrite import json_graph

#graphsage:
import time
import argparse

from graphsage_files.graphsage_simple.graphsage.dataset import Dataset 
import random
from networkx.readwrite import json_graph
from graphsage_files.graphsage_simple.graphsage.model import run_graph

#from graphsage-simple.model import run_graph

def parse_args():
    parser = argparse.ArgumentParser(description="Graphsage embedding")
    parser.add_argument('--seed', default= 42, type = int)
    parser.add_argument('--prefix', default= "graphsage_files/dataspace/bioDMLC/graphsage/bioDMLC")
    parser.add_argument('--batch_size', default = 500  , type = int)
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--max_degree', default = 5, type = int)
    parser.add_argument('--cuda', action = "store_true")
    parser.add_argument('--dim_1', default = 128, type = int)
    parser.add_argument('--dim_2', default = 128, type = int)
    parser.add_argument('--epochs',           default=100,              help='Number of epochs to train.', type=int)
    parser.add_argument('--use_random_walks',  default=False,          help="Whether to use random walk.", type=bool)
    parser.add_argument('--load_walks',        default=False,          help="Whether to load walk file.", type=bool)
    parser.add_argument('--num_walk',         default=50,              help="Number of walk from each node.", type=int)
    parser.add_argument('--walk_len',         default=5,               help="Length of each walk.", type=int)

    return parser.parse_args()
    

def load_data(prefix, supervised=False, max_degree=25, multiclass=False, use_random_walks=True, load_walks=True, num_walk=50, walk_len=5):
    
    dataset = Dataset()
    dataset.load_data(prefix=prefix, normalize=True, supervised=False, max_degree=max_degree, multiclass=multiclass, use_random_walks = use_random_walks, load_walks=load_walks, num_walk=num_walk, walk_len=walk_len)
    return dataset

def create_small_graph(max_node_label):
    karate_graph = nx.karate_club_graph()
    edges = karate_graph.edges()
    edges = np.array(edges)
    edges += max_node_label
    center = 2 + max_node_label
    return edges, center

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

def connect_two_graphs(nodes_to_concat, ori_nodes, prob_each = 0.7):
    average_deg = 3
    pseudo_edges = []
    for node in nodes_to_concat:
        if np.random.rand() < prob_each:
            to_cat = np.random.choice(ori_nodes, 3)
            pseudo_edges += [[node, ele] for ele in to_cat]
    return pseudo_edges

# for graphsage dataset creation:
def save_graph(sub_feats, G, id_map, dataset_name, dir):
        print("Saving Graph")
        num_nodes = len(G.nodes())
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

# evaluate:
def evaluate(embeddings, center1, center2, center3):
    simi = embeddings.dot(embeddings.T)

    simi_center1 = simi[center1]
    arg_sort = simi_center1.argsort()[::-1]
    print("The three centers are: ")
    print(center1, center2, center3)
    print("Seven cloest nodes to the 'center1' is: ")
    print(arg_sort[:7])
    print("The similarity values between those nodes and 'center1' is: ")
    print(simi_center1[arg_sort][:7])



if __name__ == "__main__":
    args = parse_args()
    print(args)
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path = 'bio-DM-LC.edges'
    G = read_graph(path)

    max_node_label = max(G.nodes()) + 1

    edges1, center1 = create_small_graph(max_node_label)
    G.add_edges_from(edges1)
    max_node_label = max(G.nodes()) + 1

    nodes_to_concat1 = np.array([16, 6, 4, 5, 10, 11, 12, 21, 17, 26, 14, 20, 18, 15, 22, 24, 25, 23]) + max_node_label
    pseudo_edges1 = connect_two_graphs(nodes_to_concat1, G.nodes())
    G.add_edges_from(pseudo_edges1)

    edges2, center2 = create_small_graph(max_node_label)
    G.add_edges_from(edges2)
    max_node_label = max(G.nodes()) + 1

    nodes_to_concat2 = np.array([16, 6, 4, 5, 10, 11, 12, 21, 17, 26, 14, 20, 18, 15, 22, 24, 25, 23]) + max_node_label
    pseudo_edges2 = connect_two_graphs(nodes_to_concat2, G.nodes())
    G.add_edges_from(pseudo_edges2)

    edges3, center3 = create_small_graph(max_node_label)
    G.add_edges_from(edges3)

    nodes_to_concat3 = np.array([16, 6, 4, 5, 10, 11, 12, 21, 17, 26, 14, 20, 18, 15, 22, 24, 25, 23]) + max_node_label
    pseudo_edges3 = connect_two_graphs(nodes_to_concat3, G.nodes())
    G.add_edges_from(pseudo_edges3)


    print("Number of nodes: {}, number of edges: {}, max: {}".format(len(G.nodes()), len(G.edges()), max(G.nodes())))

    num_nodes = len(G.nodes())

    ############################## for graphsage embedding ############################
    graphsage_G = nx.Graph()
    graphsage_G.add_edges_from([(str(edge[0]),str(edge[1])) for edge in G.edges()])


    features = np.ones((num_nodes,10), dtype = float) #######  Cần chỉnh sửa cách khởi tạo feature
    
    id2idx = {node:int(node) for node in graphsage_G.nodes()}
    
    save_graph(features, graphsage_G, id2idx, 'bioDMLC', 'graphsage_files/dataspace/bioDMLC')

    # graphsage_embedding:
    # load data
    graph_data = load_data(args.prefix, supervised=False, max_degree=args.max_degree, multiclass=False, \
                use_random_walks=args.use_random_walks, load_walks=False, num_walk=args.num_walk, walk_len=args.walk_len)

    embeddings, emb_model = run_graph(graph_data, args)

    evaluate(embeddings, center1, center2, center3)
