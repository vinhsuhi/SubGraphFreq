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


from models.graphsage.encoders import Encoder
from models.graphsage.aggregators import MeanAggregator
from models.graphsage.prediction import BipartiteEdgePredLayer
import pdb

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
        self.feat_data = Variable(torch.FloatTensor(feat_data), requires_grad = False)

        self.linear1 = nn.Linear(2*self.feat_data.shape[1], self.feat_data.shape[1])
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(2*self.feat_data.shape[1], self.args.dim_2)        

        self.neg_sample_size = 10
       
        self.degrees = degrees
        self.normalize_embedding = True
        self.link_pred_layer = BipartiteEdgePredLayer(is_normalized_input=self.normalize_embedding)
    
    def aggregator(self, nodes):   
        if self.args.cuda:
            #nodes = nodes.cuda()
            self.feat_data = self.feat_data.cuda()
            #self.adj_lists = self.adj_lists.cuda()
        init_nodes = nodes
        emb_hop2 = torch.zeros(len(nodes),2*(self.feat_data.shape[1]))
        
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
        for iter in range(n_iters):  ####### for iter in range(n_iters)
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

        