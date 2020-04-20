import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
from tqdm import tqdm

from models.graphsage.encoders import Encoder
from models.graphsage.aggregators import MeanAggregator
from models.graphsage.prediction import BipartiteEdgePredLayer
import pdb
import time
from copy import deepcopy

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

class SupervisedGraphSage(nn.Module):

    def __init__(self, degrees, adj_lists, feat_data,args):
        super(SupervisedGraphSage, self).__init__()
        self.args = args
        self.adj_lists = adj_lists
        # feat_data = np.concatenate((feat_data, np.zeros((1, feat_data.shape[1]))))
        self.feat_data = Variable(torch.FloatTensor(feat_data), requires_grad = False)

        self.linear1 = nn.Linear(2*self.feat_data.shape[1], self.feat_data.shape[1])
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(2*self.feat_data.shape[1], self.args.feat_dim)        

        self.neg_sample_size = 10
        self.degrees = degrees
        self.normalize_embedding = True
        self.link_pred_layer = BipartiteEdgePredLayer(is_normalized_input=self.normalize_embedding)
        self.max_degree = np.max(self.degrees)
        self.feats = torch.FloatTensor(np.array([[degrees[i]] * self.args.feat_dim for i in range(len(degrees))]))
        if self.args.cuda: 
            self.feats = self.feats.cuda()
            self.feat_data = self.feat_data.cuda()
        # import pdb
        # pdb.set_trace()
    

    def agg_one_hop(self, nodes):
        nodes = list(nodes)
        node_feats = self.feat_data[nodes]
        neighbors_feat = self.feats[nodes]
        agg = torch.cat((node_feats, neighbors_feat), dim=1)
        emb = self.linear1(agg)
        return emb


        """
        node_feats = self.feat_data[[ele for ele in nodes if ele >= 0]]
        neighbors = []
        for node in nodes:
            if node < 0:
                break
            neighbors.append(self.adj_lists[node])
        neighbor_matrix = torch.LongTensor(np.array(neighbors))
        if self.args.cuda:
            neighbor_matrix = neighbor_matrix.cuda()
        neighbor_emb = self.feat_data[neighbor_matrix].sum(dim=1)
        try:
            agg = torch.cat((node_feats, neighbor_emb), dim=1)
            emb = self.linear1(agg)
        except:
            import pdb
            pdb.set_trace()
        return emb
        """


    def old_agg(self, nodes):
        for node in init_nodes:
            nodes = set(nodes).union(self.adj_lists[node])

        
        emb_hop1 = torch.zeros(len(nodes),2*(self.feat_data.shape[1]))

        if self.args.cuda:
            emb_hop1 = emb_hop1.cuda()
            emb_hop2 = emb_hop2.cuda()
  
        new_id2idx = {node:i for i,node in enumerate(nodes)}


        
        for node in nodes:
            sum1 = 0
            for neigh in self.adj_lists[node]:
                sum1 += self.feat_data[neigh]
            node_emb = torch.cat([self.feat_data[node],sum1]) # / len(self.adj_lists[node])])
            emb_hop1[new_id2idx[node]] = node_emb
        
        emb_hop1 = self.linear1(emb_hop1)
        emb_hop1 = self.tanh(emb_hop1)


        for i,node in enumerate(init_nodes):
            sum2 = 0
            for neigh in self.adj_lists[node]:
                sum2 += emb_hop1[new_id2idx[neigh]]
            node_emb = torch.cat([self.feat_data[node],sum2]) # / len(self.adj_lists[node])])
            emb_hop2[i] = node_emb
        emb_hop = self.linear2(emb_hop2)
        return emb_hop


    def aggregator(self, nodes):
        """
        """
        # first, agg neighbors, by neighbors of neighbor
        node_feats = self.feat_data[nodes]
        neighbor_embeddings = torch.zeros((len(nodes), self.args.feat_dim))
        if self.args.cuda:
            neighbor_embeddings = neighbor_embeddings.cuda()
        for i, node in enumerate(nodes):
            neighbors_node = self.adj_lists[node]
            neighbor_emb = self.agg_one_hop(neighbors_node).sum(dim=0).flatten()
            # neighbor_embeddings.append(neighbor_emb)
            neighbor_embeddings[i] = neighbor_emb
        final_embedding = self.linear2(torch.cat((node_feats, neighbor_embeddings), dim=1))

        return final_embedding
        
        # neighbor_embeddings

        # second, agg



        # if self.args.cuda:
        """
        
        """

    def forward(self, inputs1, inputs2):        
        neg = fixed_unigram_candidate_sampler(
            num_sampled=self.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees
        )
        outputs11 = self.aggregator(inputs1.tolist())
        outputs21 = self.aggregator(inputs2.tolist())
        neg_outputs = self.aggregator(neg.tolist())

        if self.normalize_embedding:
            outputs1 = F.normalize(outputs11, dim=1)
            outputs2 = F.normalize(outputs21, dim=1)
            neg_outputs = F.normalize(neg_outputs, dim=1)
           
        return outputs1, outputs2, neg_outputs

    def loss(self, inputs1, inputs2):
        batch_size = inputs1.size()[0]
        outputs1, outputs2, neg_outputs  = self.forward(inputs1, inputs2)        
        loss = self.link_pred_layer.loss(outputs1, outputs2, neg_outputs) / batch_size
        return loss


def run_graph(graph_data,args):
    batch_size = args.batch_size

    adj_lists = graph_data.full_adj
    feat_data = graph_data.feats
    all_edges = graph_data.all_edges

    graphsage = SupervisedGraphSage(graph_data.deg, adj_lists, feat_data, args)
    if args.cuda:
        graphsage.cuda()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=args.learning_rate)

    n_iters = len(all_edges)//batch_size
     
    for epoch in range(args.epochs):
        print("Epoch {0}".format(epoch))
        np.random.shuffle(all_edges)
        for iter in tqdm(range(n_iters)):  ####### for iter in range(n_iters)
            batch_edges = torch.LongTensor(all_edges[iter*batch_size:(iter+1)*batch_size])
            optimizer.zero_grad()
            loss = graphsage.loss(batch_edges[:,0], batch_edges[:,1])
            loss.backward()
            optimizer.step()
            print("Loss: {:.4f}".format(loss.data))
    embeddings = F.normalize(graphsage.aggregator(list(range(feat_data.shape[0]))), dim = 1)
    embeddings2 = graphsage.aggregator(list(range(feat_data.shape[0])))
    embeddings = embeddings.detach().cpu().numpy()
    embeddings2 = embeddings2.detach().cpu().numpy()
    return embeddings, embeddings2

        
