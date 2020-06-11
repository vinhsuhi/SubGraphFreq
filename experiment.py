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
    parser.add_argument('--output_dim', default=6, type=int)
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
    model = FA_GCN('tanh', num_GCN_blocks, input_dim, args.output_dim)
    features = torch.FloatTensor(features)
    # adj = torch.FloatTensor(adj)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        model = model.cuda()

    num_epochs = args.epochs

    batch_size = args.batch_size

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.01)

    n_iters = len(edges)//batch_size
     
    for epoch in range(num_epochs):
        print("Epoch {0}".format(epoch))
        np.random.shuffle(edges)
        for iter in tqdm(range(n_iters)):  ####### for iter in range(n_iters)
            optimizer.zero_grad()
            batch_edges = torch.LongTensor(edges[iter*batch_size:(iter+1)*batch_size])
            output = model.forward(adj, features)
            loss = link_pred_loss(batch_edges[:, 0], batch_edges[: ,1], output, degree)
            loss.backward()
            optimizer.step()
            print("Loss: {:.4f}".format(loss.data))

    # embeddings = torch.cat(outputs, dim=1)
    
    embeddings = output.detach().cpu().numpy()
    return embeddings


def gen_data(path, kara_center, num_adds, labels=[]):
    # Nodes label is >= 1
    # Edges label is >= 0
    G, num_nodes_label, num_edges_label = read_attributed_graph(path)
    max_node_id  = max(G.nodes()) + 1
    # import pdb
    # pdb.set_trace()
    # G_mouse = read_graph('data/mouse.edges')

    nodes_to_remove = [14, 20, 18, 15, 22, 16, 5, 11, 9, 17, 12, 21] #, 3, 29, 17, 31, 32, 1, 3, 24, 8, 9]
    # nodes_to_remove = [23, 25, 28, 3, 29, 17, 31, 32, 1]
    # nodes_to_remove = []

    print("Number of nodes to be removed: {}".format(len(nodes_to_remove)))

    # 1
    edge_labels_concat = np.random.randint(0, num_edges_label, 6).tolist()
    new_node_labels = []
    new_edge_labels = []
    center1s = []
    for i in range(num_adds):
        edges, center, mapping, nodes1 = create_small_graph(max_node_id , kara_center, nodes_to_remove)
        if i == 0:
            new_node_labels = np.random.randint(0, num_nodes_label, len(nodes1)).tolist()
            new_edge_labels = np.random.randint(0, num_edges_label, len(edges)).tolist()
        center1s.append(center)
        edges = [(edges[k][0], edges[k][1], {'label': new_edge_labels[k]}) for k in range(len(edges))]
        nodes = [(nodes1[k], {'label': new_node_labels[k]}) for k in range(len(nodes1))]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        nodes_to_concat = np.array([mapping[ele] for ele in [26, 19]]) + max_node_id 
        pseudo_edges = connect_two_graphs(nodes_to_concat, G.nodes())
        G.add_edges_from([(pseudo_edges[k][0], pseudo_edges[k][1], {'label': edge_labels_concat[k]}) for k in range(len(pseudo_edges))])
        max_node_id  = max(G.nodes) + 1
        
    edge_labels_concat = np.random.randint(0, num_edges_label, 3).tolist()
    new_node_labels = []
    new_edge_labels = []
    center2s = []
    # for i in range(num_adds):
    #     edges, center, mapping, nodes2 = create_small_graph2(G_mouse, max_node_id , 9)
    #     if i == 0:
    #         new_node_labels = np.random.randint(0, num_nodes_label, len(nodes1)).tolist()
    #         new_edge_labels = np.random.randint(0, num_edges_label, len(edges)).tolist()
    #     center2s.append(center)
    #     edges = [(edges[k][0], edges[k][1], {'label': new_edge_labels[k]}) for k in range(len(edges))]
    #     G.add_nodes_from([(nodes2[k], {'label': new_node_labels[k]}) for k in range(len(nodes2))])
    #     G.add_edges_from(edges)
    #     nodes_to_concat = np.array([mapping[ele] for ele in [1]]) + max_node_id 
    #     pseudo_edges = connect_two_graphs(nodes_to_concat, G.nodes())
    #     G.add_edges_from([(pseudo_edges[k][0], pseudo_edges[k][1], {'label': edge_labels_concat[k]}) for k in range(len(pseudo_edges))])
    #     max_node_id  = max(G.nodes()) + 1

    # print("Number of nodes: {}, number of edges: {}, max: {}".format(len(G.nodes()), len(G.edges()), max(G.nodes())))
    return G, center1s, center2s, num_nodes_label, num_edges_label


def create_data_for_GCN(G, num_nodes_label):
    num_nodes = len(G.nodes)
    degree = np.array([G.degree(node) for node in G.nodes])
    edges = np.array(list(G.edges))
    # features = np.ones((num_nodes, 2))
    features = np.zeros((num_nodes, num_nodes_label))
    for node in G.nodes:
        features[node][G.nodes[node]['label']] = 1
    indexs1 = torch.LongTensor(np.array(list(G.edges)).T)
    indexs2 = torch.LongTensor(np.array([(ele[1], ele[0]) for ele in list(G.edges)]).T)
    indexs3 = torch.LongTensor(np.array([(node, node) for node in range(num_nodes)]).T)
    indexs = torch.cat((indexs1, indexs2, indexs3), dim=1)
    values = []
    for i in range(indexs1.shape[1]):
        values.append(1/(np.sqrt(G.degree(int(indexs1[0][i])) + 1) * np.sqrt(G.degree(int(indexs1[1][i])) + 1))) 
    for i in range(indexs2.shape[1]):
        values.append(1/(np.sqrt(G.degree(int(indexs2[0][i])) + 1) * np.sqrt(G.degree(int(indexs2[1][i])) + 1))) 
    for i in range(indexs3.shape[1]):
        values.append(1/(np.sqrt(G.degree(int(indexs3[0][i])) + 1) * np.sqrt(G.degree(int(indexs3[1][i])) + 1))) 
    
    values = torch.FloatTensor(np.array(values))
    adj = torch.sparse.FloatTensor(indexs, values, torch.Size([num_nodes, num_nodes]))
    return features, adj, degree, edges


def create_data_for_Graphsage(G, num_nodes_label):
    num_nodes = len(G.nodes())
    features = np.zeros((num_nodes, num_nodes_label))
    for node in G.nodes:
        features[node][G.nodes[node]['label']] = 1
    graphsage_G = nx.Graph()
    graphsage_G.add_edges_from([(str(edge[0]),str(edge[1])) for edge in list(G.edges)])
    features = np.ones((num_nodes,args.feat_dim), dtype = float) 
    id2idx = {node:int(node) for node in list(graphsage_G.nodes)}
    save_graph(features, graphsage_G, id2idx, args.data_name, args.dir)
    graph_data = load_data(args.prefix)
    return graph_data


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
        db = DBSCAN(eps=ep, min_samples=50, metric='cosine').fit(embeddings)
        labels = db.labels_
        labels = [ele + 1 for ele in labels]
        
    elif method == "LSH":
        return
    return labels


if __name__ == "__main__":
    embeddings = []
            
    args = parse_args()

    kara_center = 2
    # path = 'data/bio-DM-LC.edges'

    G, center1s, center2s, num_nodes_label, num_edges_label = gen_data(args.large_graph_path, kara_center, args.num_adds)

    print("Number of nodes: {}".format(len(G.nodes)))
    print("Number of edges: {}".format(len(G.edges)))
    # import pdb

    if args.load_embs and os.path.exists('emb.npy'):
        embeddings = np.load('emb.npy')
    else:
        if args.model == "GCN":
            features, adj, degree, edges = create_data_for_GCN(G, num_nodes_label)
            embeddings = learn_embedding(features, adj, degree, edges)
            np.save('emb.npy', embeddings)
        elif args.model == "Graphsage":
            graph_data = create_data_for_Graphsage(G, args)
            st_emb_time = time.time()
            embeddings, embeddings2 = run_graph(graph_data, args)
            print("embedding times: {:.4f}".format(time.time() - st_emb_time))

    print("Clustering...")
    st_clustering_time = time.time()
    ep = 1e-6
    while True:
        print(ep)
        labels = clustering(embeddings, args.clustering_method, ep)
        print("Clustering time: {:.4f}".format(time.time() - st_clustering_time))

        # check number of clusters
        if len(Counter(labels)) < 3:
            print("number of cluster smaller than 3")
            print("exitting...")
            exit()

        # check label of centers, if it the same, ok. But... increase epsilon
        labels_center = [labels[index] for index in center1s]
        print("Labels of centers: {}".format(labels_center))
        if (len(Counter(labels_center)) > 1) or (0 in labels_center):
            print("epsilon is too small")
            ep *= 2
            continue
        
        # if number of nodes in center's cluster is too large, then decrease epsilon
        success = evaluate(embeddings, center1s, labels, G, 'gSpan/graphdata/{}.outx'.format(args.data_name))
        if not success:
            ep /= 1.5
            continue
        else:
            print("FINAL epsilon is: {}".format(ep))
            break

    import os
    if not os.path.exists('output_graphs'):
        os.mkdir('output_graphs')

    def save_graph_to_file(G, path):
        with open(path, 'w', encoding='utf-8') as file:
            file.write('t # 1\n')
            for node in G.nodes:
                file.write('v {} {}\n'.format(node, G.nodes[node]['label']))
            for edge in G.edges:
                file.write('e {} {} {}\n'.format(edge[0], edge[1], G.edges[(edge[0], edge[1])]['label']))
        file.close()
    save_graph_to_file(G, 'output_graphs' + '/mico_10_3_added_subgraphs.lg')
    with open('output_graphs' + '/mic_10_3_added_subgraphs_centers.lg', 'w', encoding='utf-8') as file:
        for node in center1s:
            file.write('{}\n'.format(node))
    file.close()
    # print("Evaluate time: {:.4f}".format(time.time() - st_evaluate_time))

    #print("Simi between center1 and center11: {:.4f}".format(np.sum(embeddings[center1s[0]] * embeddings[center2s[0]])))


