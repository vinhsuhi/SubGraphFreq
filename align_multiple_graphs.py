import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from models.GCN import FA_GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import create_small_graph, read_graph, create_adj, connect_two_graphs, evaluate, load_data, save_graph, create_small_graph2, read_attributed_graph
from models.graphsage.model import run_graph
import argparse
from sklearn.cluster import DBSCAN
import os
from collections import Counter
import time
from models.graphsage.prediction import BipartiteEdgePredLayer



def read_graph_corpus(path, label_center_path=None):
    graphs = []
    max_node_label = 0
    with open(path, 'r', encoding='utf-8') as file:
        nodes = {}
        edges = {}
        for line in file:
            if 't' in line:
                if len(nodes) > 0:
                    graphs.append((nodes, edges))
                nodes = {}
                edges = {}
            if 'v' in line:
                data_line = line.split()
                node_id = int(data_line[1])
                node_label = int(data_line[2])
                nodes[node_id] = node_label
                if node_label > max_node_label:
                    max_node_label = node_label
            if 'e' in line:
                data_line = line.split()
                source_id = int(data_line[1])
                target_id = int(data_line[2])
                label = int(data_line[3])
                edges[(source_id, target_id)] = label
    return graphs

if __name__ == "__main__":
    graph_path = "mico.outx"
    graphs = read_graph_corpus(graph_path)
    import pdb 
    pdb.set_trace()