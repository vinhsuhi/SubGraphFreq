import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from embedding_model import FA_GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F


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

def create_adj(edges, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        adj[edge[0], edge[1]] = 1 
        adj[edge[1], edge[0]] = 1
    return adj


def connect_two_graphs(nodes_to_concat, ori_nodes, prob_each = 0.7):
    average_deg = 3
    pseudo_edges = []
    for node in nodes_to_concat:
        if np.random.rand() < prob_each:
            to_cat = np.random.choice(ori_nodes, 3)
            pseudo_edges += [[node, ele] for ele in to_cat]
    return pseudo_edges


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

    num_epochs = 10

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



def save_to_tsv(path, embeddings, labels):
    np.savetxt()


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

    features = np.ones((num_nodes, 10))
    adj = create_adj(G.edges(), num_nodes)

    embeddings = learn_embedding(features, adj)

    evaluate(embeddings, center1, center2, center3)
