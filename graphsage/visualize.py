import argparse
import numpy as np
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Convert .emb file to .tsv file for visualizing in http://projector.tensorflow.org/")
    parser.add_argument('--embed_file', default=None,  help="Embedded directory which contains embedded file (.emb) and id_map (.txt).")
    parser.add_argument('--out_file', default=None, help="Output directory.")
    parser.add_argument('--idmap_path', default=None, help="Idmap directory")    
    parser.add_argument('--file_format', default="word2vec", help="File format, choose word2vec or numpy")
    return parser.parse_args()

def load_embedding(embed_file, id_map, file_format):
    if file_format == "word2vec":
        try:
            with open(embed_file) as fp:
                descriptions = fp.readline().split()
                if len(descriptions) != 2:
                    raise Exception("Wrong format")
                num_nodes = int(descriptions[0])
                dim = int(descriptions[1])
                embeddings = np.zeros((num_nodes, dim))
                for line in fp:
                    tokens = line.split()
                    if len(tokens) != dim + 1:
                        raise Exception("Wrong format")
                    feature = np.zeros(dim)
                    for i in range(dim):
                        feature[i] = float(tokens[i+1])
                    embeddings[id_map[tokens[0]]] = feature
                fp.close()
        except Exception as e:
            print(e)
            print("The format might be wrong, consider trying --file_format flag with 'numpy' value")
            embeddings = None

    else:
        embeddings = np.load(embed_file)
        
    return embeddings

def convert_and_save(embeddings, out_file):
    """
    embeds: numpy array, shape NxD which N is the number of nodes, D is the size of embedded vector.
    id_map: dict, keys are nodes id, values are indexes.
    out_dir: directory to save output files.
    """
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    if embeddings is None:
        return
    
    with open(out_file, "w") as fp:                                
        for embedding in embeddings:            
            fp.write('\t'.join(map(str, embedding)))
            fp.write('\n')
        fp.close()   
    print("Tsv file has been saved to directory: {0}".format(out_file))

if __name__ == '__main__':
    args = parse_args()
    id_map = json.loads(open(args.idmap_path, "r").read())
    embeddings = load_embedding(args.embed_file, id_map, args.file_format)    
    convert_and_save(embeddings, args.out_file)
