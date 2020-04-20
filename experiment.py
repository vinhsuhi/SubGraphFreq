import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from models.GCN import FA_GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import create_small_graph, read_graph, create_adj, connect_two_graphs, evaluate, load_data, save_graph, create_small_graph2
from models.graphsage.model import run_graph
import argparse
from lshash import LSHash
from sklearn.cluster import DBSCAN
import os
from collections import Counter
import time
from models.graphsage.prediction import BipartiteEdgePredLayer


def parse_args():
    parser = argparse.ArgumentParser(description="Graphsage embedding")
    parser.add_argument('--model', type=str, default="GCN", help="GCN or Graphsage")
    parser.add_argument('--prefix', default= "data/bioDMLC/graphsage/bioDMLC")
    parser.add_argument('--batch_size', default = 500  , type = int)
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--cuda', action = "store_true")
    # parser.add_argument('--dim_1', default = 10, type = int)
    # parser.add_argument('--dim_2', default = 10, type = int)
    parser.add_argument('--feat_dim', default=6, type=int)
    parser.add_argument('--epochs', default=3,   help='Number of epochs to train.', type=int)
    parser.add_argument('--clustering_method', default='DBSCAN', help="choose between DBSCAN and LSH")
    parser.add_argument('--num_adds', default=3, type=int)
    parser.add_argument('--load_embs', action='store_true')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--large_graph_path', type=str, default='data/bio-DM-LC.edges')
    parser.add_argument('--dir', type=str, default='data/bioDM')

    return parser.parse_args()
    

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled


def loss_function(output, adj):
    output = F.normalize(output)
    reconstruct_adj = torch.matmul(output, output.t())
    loss = ((reconstruct_adj - adj) ** 2).mean()
    return loss


def link_pred_loss(inputs1, inputs2, embeddings, degrees):        
    neg = fixed_unigram_candidate_sampler(
        num_sampled=10,
        unique=False,
        range_max=len(degrees),
        distortion=0.75,
        unigrams=degrees
    )
    outputs1 = embeddings[inputs1.tolist()]
    outputs2 = embeddings[inputs2.tolist()]
    neg_outputs = embeddings[neg.tolist()]
    batch_size = len(outputs1)

    link_pred_layer = BipartiteEdgePredLayer(is_normalized_input=True)
    batch_isze = len(inputs1)
    loss = link_pred_layer.loss(outputs1, outputs2, neg_outputs) / batch_size
    return loss


def learn_embedding(features, adj, degree, edges):
    cuda = True
    num_GCN_blocks = 2
    input_dim = features.shape[1]
    output_dim = 2
    model = FA_GCN('tanh', num_GCN_blocks, input_dim, output_dim)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    features = torch.FloatTensor(features)
    # adj = torch.FloatTensor(adj)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        model = model.cuda()

    num_epochs = args.epochs

    # for epoch in tqdm(range(num_epochs), desc="Training..."):
    #     optimizer.zero_grad()
    #     outputs = model.forward(adj, features)
    #     loss = loss_function(outputs[-1], adj)
    #     print("loss: {:.4f}".format(loss.data))
    #     loss.backward()
    #     optimizer.step()
    batch_size = args.batch_size

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.01)

    n_iters = len(edges)//batch_size
     
    for epoch in range(num_epochs):
        print("Epoch {0}".format(epoch))
        np.random.shuffle(edges)
        for iter in tqdm(range(n_iters)):  ####### for iter in range(n_iters)
            optimizer.zero_grad()
            batch_edges = torch.LongTensor(edges[iter*batch_size:(iter+1)*batch_size])
            outputs = model.forward(adj, features)
            loss = link_pred_loss(batch_edges[:, 0], batch_edges[: ,1], outputs[-1], degree)
            loss.backward()
            optimizer.step()
            print("Loss: {:.4f}".format(loss.data))

    print("vinhsuhi1")
    embeddings = torch.cat(outputs, dim=1)
    embeddings = embeddings.detach().cpu().numpy()
    print("vinhsuhi2")
    return embeddings


def gen_data(path, kara_center, num_adds):
    G = read_graph(path)
    G = nx.relabel_nodes(G, {node: i for i, node in enumerate(list(G.nodes()))})
    G_mouse = read_graph('data/mouse.edges')
    max_node_label = max(G.nodes()) + 1

    # nodes_to_remove = [4, 5, 6, 19, 21, 31, 23, 13, 7 , 15, 22, 23, 25, 28, 3, 29, 17, 31, 32, 1, 3, 24, 8, 9]
    # nodes_to_remove = [23, 25, 28, 3, 29, 17, 31, 32, 1]
    nodes_to_remove = []

    print("Number of nodes to be removed: {}".format(len(nodes_to_remove)))

    # 1
    center1s = []
    for i in range(num_adds):
        edges, center, mapping = create_small_graph(max_node_label, kara_center, nodes_to_remove)
        center1s.append(center)
        G.add_edges_from(edges)
        max_node_label = max(G.nodes()) + 1

        nodes_to_concat = np.array([mapping[ele] for ele in [26]]) + max_node_label
        pseudo_edges = connect_two_graphs(nodes_to_concat, G.nodes())
        G.add_edges_from(pseudo_edges)

    center2s = []
    for i in range(num_adds):
        edges, center, mapping = create_small_graph2(G_mouse, max_node_label, 9)
        center2s.append(center)
        G.add_edges_from(edges)
        max_node_label = max(G.nodes()) + 1

        nodes_to_concat = np.array([mapping[ele] for ele in [1]]) + max_node_label
        pseudo_edges = connect_two_graphs(nodes_to_concat, G.nodes())
        G.add_edges_from(pseudo_edges)

    print("Number of nodes: {}, number of edges: {}, max: {}".format(len(G.nodes()), len(G.edges()), max(G.nodes())))
    return G, center1s, center2s


def create_data_for_GCN(G):
    
    num_nodes = len(G.nodes)
    degree = np.array([G.degree(node) for node in G.nodes])
    edges = np.array(list(G.edges))
    features = np.ones((num_nodes, 2))
    indexs1 = torch.LongTensor(np.array(list(G.edges)).T)
    indexs2 = torch.LongTensor(np.array([(ele[1], ele[0]) for ele in list(G.edges)]).T)
    indexs3 = torch.LongTensor(np.array([(node, node) for node in range(num_nodes)]).T)
    indexs = torch.cat((indexs1, indexs2, indexs3), dim=1)
    values = torch.FloatTensor(np.ones(indexs.shape[1]))
    adj = torch.sparse.FloatTensor(indexs, values, torch.Size([num_nodes, num_nodes]))
    # adj = nx.to_scipy_sparse_matrix(G)
    # adj = create_adj(G.edges(), num_nodes)d77
    return features, adj, degree, edges


def create_data_for_Graphsage(G, args):
    num_nodes = len(G.nodes())
    graphsage_G = nx.Graph()
    graphsage_G.add_edges_from([(str(edge[0]),str(edge[1])) for edge in list(G.edges)])
    features = np.ones((num_nodes,args.feat_dim), dtype = float) 
    id2idx = {node:int(node) for node in list(graphsage_G.nodes)}
    save_graph(features, graphsage_G, id2idx, args.data_name, args.dir)
    graph_data = load_data(args.prefix)
    return graph_data


# def load_data(data_path):
    # print("Have not implement yet")
    # return 


def save_visualize_data(embeddings, labels, method, G):
    if not os.path.exists('visualize_data'):
        os.mkdir('visualize_data')

    np.savetxt('visualize_data/{}_embeddings.tsv'.format(method), embeddings, delimiter='\t')

    with open('visualize_data/{}_labels.tsv'.format(method), 'w', encoding='utf-8') as file:
        file.write("{}\t{}\t{}\n".format('node_id', 'cluster_id', 'degree'))
        for i, lb in enumerate(labels):
            file.write("{}\t{}\t{}\n".format(i, "bucket_{}".format(lb), G.degree(i)))
    
    print("DONE saving to file!")


def clustering(embeddings, method, ep=None):
    if method == "DBSCAN":
        db = DBSCAN(eps=ep, min_samples=14, metric='cosine').fit(embeddings)
        labels = db.labels_
        labels = [ele + 1 for ele in labels]
        cter = Counter(labels)
    # if method == "DBSCAN":
    #     labels = dbscan.dbscan(embeddings.T, eps, min_points)
        
    #     import pdb
    #     pdb.set_trace()
        
    elif method == "LSH":
        model = LSHash(10, 32)
        for i in range(len(embeddings)):
            model.index(embeddings[i])
            hash_0_dict = model.hash_tables[0].storage
            key_to_index = {}
            for i, key in enumerate(hash_0_dict):
                key_to_index[key] = i
            labels = []
        for i in range(len(embeddings)):
            key = model.query(embeddings[i])[1][0]
            labels.append(key_to_index[key])

    return labels


if __name__ == "__main__":
    embeddings = []
            
    args = parse_args()

    kara_center = 2
    # path = 'data/bio-DM-LC.edges'

    G, center1s, center2s = gen_data(args.large_graph_path, kara_center, args.num_adds)

    if args.load_embs:
        embeddings2 = np.loadtxt("visualize_data/DBSCAN_embeddings.tsv", delimiter='\t')
        embeddings = F.normalize(torch.FloatTensor(embeddings2)).detach().cpu().numpy()
    else:
        if args.model == "GCN":
            features, adj, degree, edges = create_data_for_GCN(G)
            embeddings = learn_embedding(features, adj, degree, edges)
        elif args.model == "Graphsage":
            graph_data = create_data_for_Graphsage(G, args)
            st_emb_time = time.time()
            embeddings, embeddings2 = run_graph(graph_data, args)
            print("embedding times: {:.4f}".format(time.time() - st_emb_time))
    
    print("Clustering...")
    st_clustering_time = time.time()
    # for ep in [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
    for ep in [1e-6]:
        print(ep)
        labels = clustering(embeddings, args.clustering_method, ep)
        if len(Counter(labels)) < 3:
            continue
        print("Clustering time: {:.4f}".format(time.time() - st_clustering_time))
        # save_visualize_data(embeddings2, labels, args.clustering_method, G)

        st_evaluate_time = time.time()
        success = evaluate(embeddings, center1s, labels, G, 'gSpan/graphdata/{}.outx'.format(args.data_name))

        success = evaluate(embeddings, center2s, labels, G, 'gSpan/graphdata/{}.outx'.format(args.data_name))
    print("Evaluate time: {:.4f}".format(time.time() - st_evaluate_time))

    print("Simi between center1 and center11: {:.4f}".format(np.sum(embeddings[center1s[0]] * embeddings[center2s[0]])))


