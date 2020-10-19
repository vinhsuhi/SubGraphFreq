import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from models.GCN import FA_GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import create_small_graph, read_graph, create_adj, connect_two_graphs, evaluate, load_data, save_graph, create_small_graph2, create_small_graph3, create_small_graph4, read_attributed_graph
from models.graphsage.model import run_graph
import argparse
from sklearn.cluster import DBSCAN, KMeans
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
    batch_size = len(inputs1)
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
    return embeddings, model


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
    print("Number of node labels:{0}, and edge labels:{1}".format(num_nodes_label, num_edges_label))
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

    # for i in range(num_adds):
    #     edges, center, mapping, nodes1 = create_small_graph3(max_node_id)
    #     if i == 0:
    #         new_node_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    #         new_edge_labels = np.random.randint(0, num_edges_label, len(edges)).tolist()
    #     center1s.append(center)
    #     edges = [(edges[k][0], edges[k][1], {'label': new_edge_labels[k]}) for k in range(len(edges))]
    #     nodes = [(nodes1[k], {'label': new_node_labels[k]}) for k in range(len(nodes1))]
    #     G.add_nodes_from(nodes)
    #     G.add_edges_from(edges)
    #     nodes_to_concat = np.array([mapping[ele] for ele in [0]]) + max_node_id 
    #     pseudo_edges = connect_two_graphs(nodes_to_concat, G.nodes())
    #     G.add_edges_from([(pseudo_edges[k][0], pseudo_edges[k][1], {'label': edge_labels_concat[k]}) for k in range(len(pseudo_edges))])
    #     max_node_id  = max(G.nodes) + 1
        
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
        file.write("{}\t{}\t{}\t{}\n".format('node_id', 'cluster_id', 'degree','node_label'))
        for i, lb in enumerate(labels):
            file.write("{}\t{}\t{}\t{}\n".format(i, "cluster_{}".format(lb), G.degree(i), G.nodes[i]['label']))
    
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

    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
            embeddings, emb_model = learn_embedding(features, adj, degree, edges)
            np.save('emb.npy', embeddings)
        elif args.model == "Graphsage":
            graph_data = create_data_for_Graphsage(G, args)
            st_emb_time = time.time()
            embeddings, embeddings2 = run_graph(graph_data, args)
            print("embedding times: {:.4f}".format(time.time() - st_emb_time))

    print("Clustering...")
    st_clustering_time = time.time()
    #ep = 1e-6
    ep = 1e-9
    while True:
        print(ep)
        labels = clustering(embeddings, args.clustering_method, ep)
        print("Clustering time: {:.4f}".format(time.time() - st_clustering_time))

        # check number of clusters
        # if len(Counter(labels)) < 3:
        #     print("number of cluster smaller than 3")
        #     print("exitting...")
        #     exit()

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

    # Saving data for visualising
    print("Saving data for visuallise...")        
    save_visualize_data(embeddings,labels,'DBSCAN',G)        

    graphs = success
    embeddings = []
    for start_point, graph in graphs.items():
        adj = nx.adjacency_matrix(graph).todense()
        adj = torch.FloatTensor(adj)
        this_feats = features[[node for node in graph.nodes]]
        this_feats = torch.FloatTensor(this_feats)
        if True:
            adj = adj.cuda()
            this_feats = this_feats.cuda()
        embedding = emb_model(adj, this_feats)
        embedding = embedding.detach().cpu().numpy()
        embeddings.append(embedding)
    
    final_emb = np.concatenate(embeddings, axis=0)
    np.savetxt("embeddings.tsv", final_emb, delimiter="\t")
    max_len = max([len(emb) for emb in embeddings])
    kmeans = KMeans(n_clusters=max_len, random_state=0).fit(final_emb)
    kmean_labels = kmeans.labels_
    with open("k_means_labels.tsv", "w", encoding='utf-8') as file:
        # file.write("{}\t{}\n".format("graph_label", "cluster_label"))
        for i in range(len(kmean_labels)):
            file.write("{}\n".format(kmean_labels[i]))
    file.close()

    
    import pdb
    idx2id = dict()
    node_lists = [[node for node in gr.nodes] for gr in graphs.values()]
    graphs_list = list(graphs.values())
    node_atts = []
    current_based_index = 0
    for i in range(len(embeddings)):
        node_att = dict()
        for j in range(len(embeddings[i])):
            index = current_based_index + j
            cluster_label = kmean_labels[index]
            node_att[node_lists[i][j]] = {'label': graphs_list[i].nodes[node_lists[i][j]]['label'], 'cluster_label': cluster_label}
        node_atts.append(node_att)
        nx.set_node_attributes(graphs_list[i], node_att)

    
    cluster_edge_count = dict()
    for i, graph in enumerate(graphs_list):
        for edge in graph.edges(data=True):
            node_0, node_1, edge_label = edge[0], edge[1], edge[2]['label']
            node_0_cluster = graph.nodes[node_0]['cluster_label']
            node_1_cluster = graph.nodes[node_1]['cluster_label']
            node_0_label = graph.nodes[node_0]['label']
            node_1_label = graph.nodes[node_1]['label']
            cluster_pair = sorted([node_0_cluster, node_1_cluster]) + [edge_label] + sorted([node_0_label, node_1_label])
            cluster_pair = "_".join([str(ele) for ele in cluster_pair])
            if cluster_pair not in cluster_edge_count:
                cluster_edge_count[cluster_pair] = {i: sorted([node_0, node_1])}
            else:
                cluster_edge_count[cluster_pair][i] = sorted([node_0, node_1])
    
    pdb.set_trace()



    # for i in range(len(final_emb)):
    #     graph_index = -1
    #     sum_len = 0
    #     while sum_len <= i:
    #         graph_index += 1
    #         sum_len += len(embeddings[graph_index])
    #     sum_len -= len(embeddings[graph_index])
    #     cluster_id = kmean_labels[i]


            

    exit()
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


