import _pickle as pickle 
import pdb
import numpy as np
import json
import networkx as nx
import os
from networkx.readwrite import json_graph

def save_graph(sub_feats, G, id_map, dataset_name, dir):
        print("Saving Graph")
        num_nodes = len(G.nodes())
        rand_indices = np.random.permutation(num_nodes)
        train = rand_indices[:int(num_nodes * 0.81)]
        val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
        test = rand_indices[int(num_nodes * 0.9):]
        res = json_graph.node_link_data(G)    
        res['nodes'] = [
            {
                'id': node['id'],
                'val': id_map[str(node['id'])] in val,
                'test': id_map[str(node['id'])] in test
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

G = nx.Graph()
with open(r'C:\Users\LENOVO\Desktop\GraphPapers\SubGraphFreq\bio-DM-LC.edges','r') as f:
    file = f.readlines()
    for line in file:
        G.add_edge(line.replace('\n','').split()[0],line.replace('\n','').split()[1])
    
    

id2idx = {node:i for (i,node) in enumerate(G.nodes())}
feats = np.ones((len(id2idx),len(id2idx)))
save_graph(feats, G, id2idx, 'citeseer', 'dataspace/citeseer')