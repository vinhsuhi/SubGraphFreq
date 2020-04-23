import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from embedding_model import FA_GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F
from networkx.readwrite import json_graph
from models.graphsage.dataset import Dataset 
from collections import Counter
import os
import json
import networkx.algorithms.isomorphism as iso

def create_small_graph(max_node_label, kara_center=2, nodes_to_remove=[]):
    karate_graph = nx.karate_club_graph()
    karate_graph.remove_nodes_from(nodes_to_remove)
    mapping = {node: i for i, node in enumerate(karate_graph.nodes())}
    center = mapping[kara_center] + max_node_label
    karate_graph = nx.relabel_nodes(karate_graph, mapping)
    edges = karate_graph.edges()
    edges = np.array(edges)
    edges += max_node_label
    nodes = np.array(list(karate_graph.nodes)) + max_node_label
    return edges, center, mapping, nodes.tolist()


def create_small_graph2(graph, max_node_label, center):
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    center = mapping[center] + max_node_label
    graph = nx.relabel_nodes(graph, mapping)
    edges = graph.edges()
    edges = np.array(edges)
    edges += max_node_label
    nodes = np.array(list(graph.nodes)) + max_node_label
    return edges, center, mapping, nodes.tolist()



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


def read_attributed_graph(path):
    nodes = []
    edges = []
    node_feats = set()
    edge_feats = set()
    # max_node_feat = 0
    # max_edge_feat = 0
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            if 'v' in line:
                nodes.append((int(data_line[1]), {'label': int(data_line[2])}))
                node_feats.add(int(data_line[2]))
            elif 'e' in line:
                edges.append((int(data_line[1]), int(data_line[2]), {'label': int(data_line[3])}))
                edge_feats.add(int(data_line[3]))
    
    node_feats = list(node_feats)
    edge_feats = list(edge_feats)

    new_node_feats = {feat: i for i, feat in enumerate(node_feats)}
    new_edge_feats = {feat: i for i, feat in enumerate(edge_feats)}

    G = nx.Graph() 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    for node in G.nodes:
        G.nodes[node]['label'] = new_node_feats[G.nodes[node]['label']]
    
    for edge in G.edges:
        G.edges[edge]['label'] = new_edge_feats[G.edges[edge]['label']]
    
    num_node_labels = len(new_node_feats)
    num_edge_labels = len(new_edge_feats)

    nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes)})
    return G, num_node_labels, num_edge_labels
        


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
        to_cat = np.random.choice(ori_nodes, 3)
        pseudo_edges += [[node, ele] for ele in to_cat]
    return pseudo_edges


def evaluate(embeddings, centers, cluster_labels, Graph, file_name):
    Threshold = 80
    print("-"*100)
    
    labels_counter = Counter(labels)
    
    labels_center = [labels[index] for index in centers]
    labels_center_count = Counter(labels_center)
    if len(labels_center_count) > 1:
        return 0
    print("clusering results, number of clusters: {}".format(len(labels_counter)))
    print(labels_counter)
    print("labels of center nodes: {}".format(labels_center_count))
    for key, value in labels_center_count.items():
        print("Cluster {} has {} centers over {} nodes".format(key, value, labels_counter[key]))

    label = labels[centers[0]]
    points_in_label = [index for index in range(len(labels)) if labels[index] == label]
    count_att_labels = Counter([Graph[node]['label'] for node in points_in_label])
    points_in_label = [node for node in points_in_label if count_att_labels[Graph[node]['label']] > Threshold]
    att_label_set = set([Graph[node]['label'] for node in points_in_label])
    # count_att_labels = Counter([Graph[node]['label'] for node in points_in_label])
    # num_unique_att = len(count_att_labels)
    if labels_counter[labels[centers[0]]] > 5000:
        print("too large")
        return 0
    
    results = save_subgraph(Graph, points_in_label, centers, file_name, list(att_label_set))
    return 1



def save_subgraph(Graph, points_in_label, true_labels, file_name, att_label_set):
    # node_feats += 1
    subgraphs = {}
    att_label_dict = {ele: i for i, ele in enumerate(att_label_set)}
    for start_node in points_in_label:
        subgraph_depth = get_subgraph(Graph, start_node,2)
        if len(subgraph_depth.nodes) > 300:
            continue
        subgraphs[start_node] = subgraph_depth

    count = 0
    groundtruth = []
    with open(file_name, 'w', encoding='utf-8') as file:
        for key, value in subgraphs.items():
            if key in true_labels:
                groundtruth.append(count)
            file.write('t # {}\n'.format(count))
            count += 1
            id2idx = {node: i for i, node in enumerate(list(value.nodes()))}
            for node in id2idx:
                the_start_label = Graph.nodes[node]['label']
                if the_start_label not in att_label_dict:
                    the_start_label += len(att_label_dict)
                else:
                    the_start_label = att_label_dict[the_start_label]
                if node == key:
                    file.write('v {} {}\n'.format(id2idx[node], the_start_label))
                else:
                    file.write('v {} {}\n'.format(id2idx[node], the_start_label))
                
            for edge in value.edges():
                file.write('e {} {} {}\n'.format(id2idx[edge[0]], id2idx[edge[1]], Graph.edges[(edge[0], edge[1])]['label']))
    file.close()

    with open(file_name + "labels_graph", 'w', encoding='utf-8') as file:
        for gt in groundtruth:
            file.write('{}\t'.format(gt))
    file.close()
    with open(file_name + "att_label_center", 'w', encoding='utf-8') as file:
        for label in att_label_set:
            file.write('{}\n'.format(label))
    file.close()
    # with open(file_name + "true_label", 'w', encoding='utf-8') as file:
    #     for label in true_labels:
    #         file.write('{}\n'.format(label))
    file.close()



        
    # Groups = {1: {'points': [points_in_label[0]], 'num_nodes': len(subgraphs[points_in_label[0]].nodes())}}
    # for i in tqdm(range(1, len(points_in_label))):
    #     create_new = True
    #     # for k, gr in enumerate(Groups):
    #     for key, value in Groups.items():
    #         gr = Groups[key]
    #         gr_repre_subgraph = subgraphs[gr['points'][0]]
    #         if nx.is_isomorphic(gr_repre_subgraph, subgraphs[points_in_label[i]]):
    #             Groups[key]['points'].append(points_in_label[i])
    #             create_new = False
    #             break
    #     if create_new:
    #         # Groups.append([i])
    #         Groups[len(Groups) + 1] = {'points': [points_in_label[i]], 'num_nodes': len(subgraphs[points_in_label[i]].nodes())}
    # Group_depth[depth] = Groups


def get_bfs_results(Graph, points_in_label):
    Group_depth = {}
    for depth in range(1, 3):
        subgraphs = {}
        for start_node in points_in_label:
            subgraph_depth = get_subgraph(Graph, start_node, depth)
            subgraphs[start_node] = subgraph_depth
        Groups = {1: {'points': [points_in_label[0]], 'num_nodes': len(subgraphs[points_in_label[0]].nodes())}}
        for i in tqdm(range(1, len(points_in_label))):
            create_new = True
            # for k, gr in enumerate(Groups):
            for key, value in Groups.items():
                gr = Groups[key]
                gr_repre_subgraph = subgraphs[gr['points'][0]]
                if nx.is_isomorphic(gr_repre_subgraph, subgraphs[points_in_label[i]]):
                    Groups[key]['points'].append(points_in_label[i])
                    create_new = False
                    break
            if create_new:
                # Groups.append([i])
                Groups[len(Groups) + 1] = {'points': [points_in_label[i]], 'num_nodes': len(subgraphs[points_in_label[i]].nodes())}
        Group_depth[depth] = Groups
    return Group_depth

def get_bfs_results_new(Graph, points_in_label, minS):
    """
    ban dau co 3 node root
    3 stacks
    lam mot vong lap, tai moi vong lap thi pop ra mot nut, cung thi luu lai, chua ket luan
    degree khac nhau: tach ra thanh cac cum nho hon, chi giu lai nhung cum lon hoi minSub
    3 thang cung degree thi moi xet den neighbor 

    * Trong mot stack co vai dinh co degree giong nhau:
    * Cac subgraph co duoc overlap khong:
    * Cac 

    # 1 Cac node in label phai cung degree
    # 2 
    """
    neighborss = []
    for node in points_in_label:
        sorted_neighbors = sorted_neighbors_by_degree(node, Graph)
        neighborss.append(sorted_neighbors)
    
    # for neighbors in neighborss:
    firsts = [ele[0] for ele in neighborss]

    cluster_by_degree = get_cluster_by_degree(firsts, Graph)

    pass

def get_cluster_by_degree(list_node, Graph):
    """
    TODO: DONE!
    """
    degree_cluster = {Graph.degree(list_node[0]): [list_node[0]]}
    for i in range(len(list_node)):
        if i == 0:
            continue
        degree_node = Graph.degree(list_node[i])
        if degree_node not in degree_cluster:
            degree_cluster[degree_node] = [list_node[i]]
        else:
            degree_cluster[degree_node].append(list_node[i])
    cluster_by_degree = list(degree_cluster.values())
    return cluster_by_degree


def sorted_neighbors_by_degree(node, Graph):
    neighbors = Graph.neighbors(node)
    neighbors = np.array(neighbors)
    degree = np.array([Graph.degree(n) for n in neighbors])
    return neighbors[np.argsort(degree)]

 

def get_subgraph(Graph, start_node, depth):
    nodes = set([start_node])
    for i in range(depth):
        nodes_i = set()
        for node in nodes:
            neighbors = Graph.neighbors(node)
            nodes_i.update(neighbors)
        nodes.update(nodes_i)
    return Graph.subgraph(nodes)



def jaccard_distance(list_1, list_2):
    set1 = set(list_1)
    set2 = set(list_2)
    common = set1.intersection(set2)
    uni = set1.union(set2)
    return len(common) / len(uni)


def load_data(prefix):
    dataset = Dataset()
    dataset.load_data(prefix=prefix, normalize=True)
    return dataset


def save_graph(sub_feats, G, id_map, dataset_name, dir):
    print("Saving Graph")
    num_nodes = len(list(G.nodes))
    rand_indices = np.random.permutation(num_nodes)
    res = json_graph.node_link_data(G)    
    res['nodes'] = [
        {
            'id': node['id']
        }
        for node in res['nodes']]
                    
    res['links'] = [
        {
            'source': link['source'],
            'target': link['target']
        }
        for link in res['links']]

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")
                
    with open(dir + "/graphsage/" + dataset_name + "-G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + dataset_name + "-id_map.json", 'w') as outfile:
        json.dump(id_map, outfile)
        
    print("GraphSAGE format stored in {0}".format(dir + "/graphsage/"))
    np.save(dir + "/graphsage/" + dataset_name + '-feats.npy',sub_feats)
