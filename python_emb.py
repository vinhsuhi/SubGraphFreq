from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import json

import torch
import torch.nn.functional as F
# from graphsage.unsupervised_train import train, load_data, get_embedding
from graphsage_simple.graphsage.dataset import Dataset 
import networkx as nx
import argparse
import os
import pdb
import random
from graphsage.unsupervised_models import UnsupervisedGraphSage
from graphsage.neigh_samplers import UniformNeighborSampler
from networkx.readwrite import json_graph
from graphsage_simple.graphsage.model import run_graph

#from graphsage-simple.model import run_graph
def load_data(prefix, supervised=False, max_degree=25, multiclass=False, use_random_walks=True, load_walks=True, num_walk=50, walk_len=5):
    
    dataset = Dataset()
    dataset.load_data(prefix=prefix, normalize=True, supervised=False, max_degree=max_degree, multiclass=multiclass, use_random_walks = use_random_walks, load_walks=load_walks, num_walk=num_walk, walk_len=walk_len)
    return dataset



def parse_args():
    parser = argparse.ArgumentParser(description="Query embedding")
    parser.add_argument('--embedding_model', default="unsup/graphsage_mean_0.001000/")
    parser.add_argument('--seed', default= 42, type = int)
    parser.add_argument('--prefix', default= "dataspace/graph/ppi/graphsage/ppi")
    # parser.add_argument('--prefix_subgraph', default= "dataspace/graph/ppi/graphsage/sub_graph")
    parser.add_argument('--batch_size', default = 12  , type = int)
    parser.add_argument('--print_every', default = 10,  type = int)
    parser.add_argument('--identity_dim', default = 0,  type = int)
    parser.add_argument('--samples_1', default = 8, type = int)
    parser.add_argument('--samples_2', default = 4, type = int) 
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--base_log_dir', default = "visualize/karate")
    parser.add_argument('--neg_sample_size', default = 2, type = int)
    parser.add_argument('--max_degree', default = 5, type = int)
    # parser.add_argument('--cuda', default = True, type = bool)
    parser.add_argument('--cuda', action = "store_true")
    parser.add_argument('--dim_1', default = 128, type = int)
    parser.add_argument('--dim_2', default = 128, type = int)
    parser.add_argument('--dir', default = './dataspace/graph/')
    parser.add_argument('--random_delete_nodes', default = 0.0, type = float)



    parser.add_argument('--model',            default='graphsage_mean',help='Model names. See README for possible values.')
    parser.add_argument('--multiclass',       default=False,           help='Whether use 1-hot labels or indices.', type=bool)
    parser.add_argument('--concat',           default=True,            help='Whether to concat', type=bool)
    parser.add_argument('--epochs',           default=1,              help='Number of epochs to train.', type=int)
    parser.add_argument('--dropout',          default=0.0,             help='Dropout rate (1 - keep probability).', type=float)
    parser.add_argument('--weight_decay',     default=0.0,             help='Weight for l2 loss on embedding matrix.', type=float)
    parser.add_argument('--samples_3',        default=0,               help='Number of users samples in layer 3. (Only for mean model)', type=int)
    parser.add_argument('--random_context',   default=False,           help='Whether to use random context or direct edges', type=bool)
    parser.add_argument('--max_total_steps',  default=10**10,          help="Maximum total number of iterations", type=int)
    parser.add_argument('--validate_iter',    default=5000,            help="How often to run a validation minibatch.", type=int)
    parser.add_argument('--validate_batch_size', default=256,          help="How many nodes per validation sample.", type=int)
    parser.add_argument('--save_embeddings',  default=False,           help="Whether to save val embeddings.", type=bool)
    parser.add_argument('--use_pre_train',     default=False,          help="Whether to use pretrain embeddings.", type=bool)
    parser.add_argument('--use_random_walks',  default=False,          help="Whether to use random walk.", type=bool)
    parser.add_argument('--load_walks',        default=False,          help="Whether to load walk file.", type=bool)
    parser.add_argument('--num_walk',         default=50,              help="Number of walk from each node.", type=int)
    parser.add_argument('--walk_len',         default=5,               help="Length of each walk.", type=int)
    parser.add_argument('--load_embedding_samples_dir',  default=None, help="Whether to load embedding samples.")
    parser.add_argument('--save_embedding_samples',  default=False,    help="Whether to save embedding samples", type=bool)
    parser.add_argument('--load_adj_dir',     default=None,            help="Adj dir load")
    parser.add_argument('--load_model_dir',   default=None,            help="model dir load")
    parser.add_argument('--save_model',       default=False,           help="Whether to save model", type=bool)
    parser.add_argument('--no_feature',       default=False,           help='whether to use features')
    parser.add_argument('--normalize_embedding', default=True,        help='whether to use features')
    parser.add_argument('--max_subgraph_nodes', default=100, type=int,        help='whether to use features')

    return parser.parse_args()



class GraphQuery():
    def __init__(self, ori_graph_data, ori_emb_model, args):
        self.args = args

        # origraph
        self.ori_graph_data = ori_graph_data
        self.ori_graph = ori_graph_data.G
        self.id_map = ori_graph_data.id_map
        self.feats = ori_graph_data.feats
        self.ori_emb_model = ori_emb_model
        #if os.path.exists(self.args.prefix + "-feats.npy"):
        self.raw_feats = ori_graph_data.raw_feats
        # subgraph
        self.sub_graph = None
        self.sub_id_map = None
        self.sub_feats = None
        self.sub_adj = None
        self.sub_degree = None


    def create_subgraph(self, num_nodes):
        print("Generating Subgraph...")
        nodes = []
        del_nodes = []
        index = 0 
        failed_add = 0
        source_node = random.choice(self.ori_graph.nodes())
        self.source_node = source_node
        nodes.append(source_node)
        while num_nodes > len(nodes):
            #print('gotten subgraph nodes: {}'.format(len(nodes)))
            len_nodes = len(nodes)
            if index >= len(nodes):
                break
            for i in range(index, len(nodes)):
                for node in self.ori_graph.neighbors(nodes[i]):
                    if node not in nodes:
                        if np.random.uniform(0,1) > self.args.random_delete_nodes:
                            nodes.append(node)
            index = len_nodes
        
        # remove nodes exceed num_nodes:
        for i in range(index, len(nodes)):
            if np.random.uniform(0,1) > (num_nodes - index) / (len(nodes) - index):
                del_nodes.append(nodes[i])
        nodes = [node for node in nodes if node not in del_nodes]
        self.sub_graph = self.ori_graph.subgraph(nodes)
        print('num subgraph nodes: {}'.format(len(nodes)))

        # create sub_graph_map:
        self.sub_id_map = {id:i for i, id in enumerate(self.sub_graph.nodes())}
        sub_id_map_inverse = {v:k for k, v in self.sub_id_map.items()}
        self.sub_feats = np.zeros((len(self.sub_id_map), self.feats.shape[1]))
        self.sub_raw_feats = np.zeros((len(self.sub_id_map), self.raw_feats.shape[1]))

        for i in range(len(self.sub_id_map)):
            id = sub_id_map_inverse[i]
            old_index = self.id_map[id]
            self.sub_feats[i] = self.feats[old_index]
            self.sub_raw_feats[i] = self.raw_feats[old_index]
        ## create adj:
        #self.sub_adj = -1*np.ones((len(self.sub_id_map)+1, self.args.max_degree)).astype(int)
        self.all_sub_adj = {}
        for nodeid in self.sub_graph.nodes():
            neighbors = np.array([self.sub_id_map[neighbor] for neighbor in self.sub_graph.neighbors(nodeid)])
            self.all_sub_adj[self.sub_id_map[nodeid]] = set(neighbors)
            #if len(neighbors) > self.args.max_degree:
            #    neighbors = np.random.choice(neighbors, self.args.max_degree, replace=False)
            #elif len(neighbors) < self.args.max_degree:
            #    neighbors = np.random.choice(neighbors, self.args.max_degree, replace=True)
            #self.sub_adj[self.sub_id_map[nodeid], :] = neighbors

        self.sub_raw_feats = torch.FloatTensor(self.sub_raw_feats)
        #self.sub_adj = torch.LongTensor(self.sub_adj)
        if self.args.cuda:
            self.all_sub_adj = self.all_sub_adj.cuda()
            self.sub_raw_feats = self.sub_raw_feats.cuda()
        self.sub_degree = np.array([self.sub_graph.degree(node) for node in self.sub_graph.nodes()])

        return self.sub_graph, self.sub_id_map, self.sub_raw_feats, self.all_sub_adj, self.sub_degree
    
    # def jaccard_sim(self,list1,list2):

    #     intersection = []
    #     union = []
    #     #pdb.set_trace()
    #     for i in range(len(list1)):
    #         if list1[i] in list2 and list1[i] not in intersection:
    #             intersection.append(list1[i])

    #     for i in range(len(list1)):
    #         and vec not in union:
    #         union.append(vec)
    #     for vec in list2 and vec not in union:
    #         union.append(vec)
    #     pdb.set_trace()

    #     return (len(intersection)-1) / (len(union)-1)
        
    def query_nodes(self, embeddings, embedding_subgraph):
        centroid_node = self.get_centroid(self.sub_graph)
        # centroid_node = self.source_node
        print('max degree node: {}'.format(centroid_node))
        print('source node: {}'.format(self.source_node))
        old_index = self.id_map[centroid_node]
        new_index = self.sub_id_map[centroid_node]

        distance_matrix = ((embeddings - embedding_subgraph[new_index])**2).sum(axis = 1)
        arg_sort_distance = np.argsort(distance_matrix)
    
        print('num nodes source_graph, sub_graph: {}  {}'.format(len(embedding_subgraph),len(embeddings)))
        print('mean distance: {:.4f}'.format(np.mean(distance_matrix)))
        print('distance anchor: {:.4f}'.format(distance_matrix[old_index]))
        print('min, max distance: {:.4f} {:.4f}'.format(distance_matrix[arg_sort_distance[0]], distance_matrix[arg_sort_distance[-1]]))

        for i in range(len(arg_sort_distance)):
            if arg_sort_distance[i] == old_index:
                print('rank anchor: {}'.format(i+1))
                print('accuracy: {:.4f}'.format(1/(i+1)))
                return i+1

    def get_centroid(self, sub_graph):
        """
        return node with max degree of subgraph
        """
        centroid_node = 0
        max_degree = 0
        for node in sub_graph.nodes():
            if sub_graph.degree(node) > max_degree:
                max_degree = sub_graph.degree(node)
                centroid_node = node
        return centroid_node
    
    def embedding_subgraph(self): ### consider args
        self.ori_emb_model.feat_data = self.sub_raw_feats
        self.ori_emb_model.adj_lists = self.all_sub_adj
        self.ori_emb_model.degrees = self.sub_degree
        # model.sample_fn = UniformNeighborSampler(train_adj)
        return self.ori_emb_model.aggregator(list(range(self.sub_raw_feats.shape[0])))

def normalize(embedding):
    return embedding / (np.sqrt((embedding**2).sum(axis = 1)).reshape(len(embedding),1))

    

if __name__ == '__main__':
    # get arguments
    args = parse_args()
    print(args)
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    ori_graph_data = load_data(args.prefix, supervised=False, max_degree=args.max_degree, multiclass=False, \
                use_random_walks=args.use_random_walks, load_walks=False, num_walk=args.num_walk, walk_len=args.walk_len)

    ori_graph_emb, ori_emb_model = run_graph(ori_graph_data, args)

    # Graphsage unsupervised train for original graph => Embedding and Model
    # _, _, ori_graph_emb, ori_emb_model = train(ori_graph_data, args)

    # query_machine definition
    for del_nodes in [0.1,0.2]:
        for max_nodes in [50,100]:
            print('del: {}, max_nodes: {}'.format(del_nodes,max_nodes))
            args.max_subgraph_nodes = max_nodes
            args.random_delete_nodes = del_nodes
            query_machine = GraphQuery(ori_graph_data, ori_emb_model, args)
            # create subgraph
            query_machine.create_subgraph(args.max_subgraph_nodes)
            # embedding subgraph
            embedding_subgraph = query_machine.embedding_subgraph()
            embedding_subgraph = embedding_subgraph.detach().cpu().numpy()
            
            accuracy = query_machine.query_nodes(normalize(ori_graph_emb), normalize(embedding_subgraph))


   

