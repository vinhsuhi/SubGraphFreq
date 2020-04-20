from __future__ import print_function
from pathlib import Path

import numpy as np
import json
import sys
import os
import pdb

import networkx as nx
from networkx.readwrite import json_graph
import pdb
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

class Dataset:

    N_WALKS = 50
    WALK_LEN = 5

    def __init__(self):
        self.prefix = None
        self.G = None
        self.feats = None
        self.all_edges = None
        self.conversion = None
        self.full_adj = None
        self.deg = None


    def load_data(self, prefix, normalize=True, train_all_edge=True):
        self.prefix = prefix
        self.read_data(prefix)
        self.preprocess_data(prefix, normalize, train_all_edge)


    def read_data(self, prefix):
        print("-----------------------------------------------")
        print("Loading data:")

        ## Load graph data
        print("Loading graph data from {0}".format(prefix + "-G.json"))
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)

        if isinstance(list(G.nodes)[0], int):
            def conversion(n): return int(n)
        else:
            def conversion(n): return n

        self.G = G
        self.conversion = conversion
        print("File loaded successfully")

        ## Load feature data
        print("Loading feature from {0}".format(prefix + "-G.json"))
        if os.path.exists(prefix + "-feats.npy"):
            self.feats = np.load(prefix + "-feats.npy")
            print("File loaded successfully")
        else:
            print("No features present.. Only identity features will be used.")

        id_map = json.load(open(prefix + "-id_map.json"))
        id_map = {conversion(k): int(v) for k, v in id_map.items()}
        idx2id = {v: k for k, v in id_map.items()}
        self.id_map = id_map
        self.idx2id = idx2id


    def preprocess_data(self, prefix, normalize=True, train_all_edge=False):
        G = self.G
        if G == None:
            raise Exception("Data hasn't been load")

        print("Loaded data.. now preprocessing..")

        self.nodes_ids = np.array(list(G.nodes()))
        self.all_edges = np.array([[self.id_map[node1],self.id_map[node2]] for (node1,node2) in list(self.G.edges)])

        self.construct_all_deg()

        self.construct_all_adj()

        print("Preprocessing finished, graph info:")
        print(nx.info(G))


    def construct_all_deg(self):
        self.deg = np.zeros((len(self.id_map),)).astype(int)
        for nodeid in self.nodes_ids:
            neighbors = np.array([self.id_map[neighbor] for neighbor in self.G.neighbors(nodeid)])
            self.deg[self.id_map[nodeid]] = len(neighbors)


    def construct_all_adj(self):
        self.full_adj = {}
        for nodeid in self.nodes_ids:
            neighbors = [self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ]
            if len(neighbors) < 300:
                neighbors += [-1] * (300 - len(neighbors))
            neighbors = neighbors[:300]
            self.full_adj[self.id_map[nodeid]] = neighbors     


if __name__ == "__main__":
    data_dir = sys.argv[1]
    preprocessor = Dataset()
    preprocessor.load_data(prefix = data_dir, supervised=False)
    preprocessor.run_random_walks(out_file = data_dir + "_walks.txt")
