import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from models.GCN import FA_GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import create_small_graph, read_graph, create_adj, connect_two_graphs, evaluate, load_data, save_graph
from models.graphsage.model import run_graph
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Graphsage embedding")
    parser.add_argument('--model', type=str, default="GCN", help="GCN or Graphsage")
    parser.add_argument('--prefix', default= "data/bioDMLC/graphsage/bioDMLC")
    parser.add_argument('--batch_size', default = 500  , type = int)
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--cuda', action = "store_true")
    parser.add_argument('--dim_1', default = 64, type = int)
    parser.add_argument('--dim_2', default = 64, type = int)
    parser.add_argument('--epochs', default=3,   help='Number of epochs to train.', type=int)

    return parser.parse_args()
    


def loss_function(output, adj):
    output = F.normalize(output)
    reconstruct_adj = torch.matmul(output, output.t())
    loss = ((reconstruct_adj - adj) ** 2).mean()
    return loss


def learn_embedding(features, adj):
    cuda = False
    num_GCN_blocks = 2
    input_dim = features.shape[1]
    output_dim = 20
    model = FA_GCN('tanh', num_GCN_blocks, input_dim, output_dim)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        model = model.cuda()

    num_epochs = 3

    for epoch in tqdm(range(num_epochs), desc="Training..."):
        optimizer.zero_grad()
        outputs = model.forward(adj, features)
        loss = loss_function(outputs[-1], adj)
        print("loss: {:.4f}".format(loss.data))
        loss.backward()
        optimizer.step()
    
    embeddings = torch.cat(outputs, dim=1)
    embeddings = embeddings.detach().cpu().numpy()
    return embeddings


def gen_data(path, kara_center):
    G = read_graph(path)
    max_node_label = max(G.nodes()) + 1

    # nodes_to_remove = [4, 5, 6, 19, 21, 31, 23, 13, 7 , 15, 22, 23, 25, 28, 3, 29, 17, 31, 32, 1, 3, 24, 8, 9]
    # nodes_to_remove = [23, 25, 28, 3, 29, 17, 31, 32, 1]
    nodes_to_remove = []

    print("Number of nodes to be removed: {}".format(len(nodes_to_remove)))

    edges1, center1, mapping = create_small_graph(max_node_label, kara_center, nodes_to_remove)
    G.add_edges_from(edges1)
    max_node_label = max(G.nodes()) + 1

    nodes_to_concat1 = np.array([mapping[ele] for ele in [3]]) + max_node_label
    pseudo_edges1 = connect_two_graphs(nodes_to_concat1, G.nodes())
    G.add_edges_from(pseudo_edges1)

    edges2, center2, mapping = create_small_graph(max_node_label, kara_center, nodes_to_remove)
    G.add_edges_from(edges2)
    max_node_label = max(G.nodes()) + 1

    nodes_to_concat2 = np.array([mapping[ele] for ele in [3]]) + max_node_label
    pseudo_edges2 = connect_two_graphs(nodes_to_concat2, G.nodes())
    G.add_edges_from(pseudo_edges2)

    edges3, center3, mapping = create_small_graph(max_node_label, kara_center, nodes_to_remove)
    G.add_edges_from(edges3)

    nodes_to_concat3 = np.array([mapping[ele] for ele in [3]]) + max_node_label
    pseudo_edges3 = connect_two_graphs(nodes_to_concat3, G.nodes())
    G.add_edges_from(pseudo_edges3)
    print("Number of nodes: {}, number of edges: {}, max: {}".format(len(G.nodes()), len(G.edges()), max(G.nodes())))
    return G, center1, center2, center3


def create_data_for_GCN(G):
    num_nodes = len(G.nodes())
    features = np.ones((num_nodes, 10))
    adj = create_adj(G.edges(), num_nodes)
    return features, adj


def create_data_for_Graphsage(G):
    num_nodes = len(G.nodes())
    graphsage_G = nx.Graph()
    graphsage_G.add_edges_from([(str(edge[0]),str(edge[1])) for edge in list(G.edges)])
    features = np.ones((num_nodes,10), dtype = float) 
    id2idx = {node:int(node) for node in list(graphsage_G.nodes)}
    save_graph(features, graphsage_G, id2idx, 'bioDMLC', 'data/bioDMLC')
    graph_data = load_data(args.prefix)
    return graph_data


if __name__ == "__main__":
    args = parse_args()

    kara_center = 2
    path = 'data/bio-DM-LC.edges'

    G , center1, center2, center3 = gen_data(path, kara_center)

    if args.model == "GCN":
        features, adj = create_data_for_GCN(G)
        embeddings = learn_embedding(features, adj)
    elif args.model == "Graphsage":
        graph_data = create_data_for_Graphsage(G)
        embeddings = run_graph(graph_data, args)
    success = evaluate(embeddings, center1, center2, center3)
