# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
vecmap_dir = os.path.dirname(current_dir) + "/vecmap"
sys.path.insert(0, vecmap_dir)

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys
import pdb
import networkx as nx
from networkx.readwrite import json_graph
import json


BATCH_SIZE = 500


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--prefix_source', default="example_data/graph/graphsage/graph", help="Prefix source")
    parser.add_argument('--prefix_target', default="example_data/graph/graphsage/graph", help="Prefix target")
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls', 'topk'],
        help='the retrieval method (\
        nn: standard nearest neighbor; \
        invnn: inverted nearest neighbor; \
        invsoftmax: inverted softmax; \
        csls: cross-domain similarity local scaling;\
        topk: ...)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()
    print(args)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    print(srcfile)
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    #  for i in range(len(x)):
    #    print(np.sum(np.square(x[i] - z[0])))
        # print(np.dot(x[i], z[i]))

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)

    src = list(src2trg.keys())
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

    # Find translations
    translation = collections.defaultdict(int)
    if args.retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = x[src[i:j]].dot(z.T)
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif args.retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]

    elif args.retrieval == 'topk':
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2 * x[src[i:j]].dot(z.T) - knn_sim_bwd
            nn = similarities.argsort(axis=1)[:,::-1][:,:args.neighborhood].tolist()
            # nn = similarities.argmax(axis=1).tolist()
            for k in range(j - i):
                translation[src[i + k]] = nn[k]

    # Compute accuracy
    count = 0
    #print(translation)


    if args.retrieval == 'topk':
        # load G
        G_data = json.load(open(args.prefix_target + "-G.json", "r"))
        G = json_graph.node_link_graph(G_data)

        source_features, target_features = None, None
        if os.path.exists(args.prefix_source + "-feats.npy") and os.path.exists(args.prefix_target+ "-feats.npy"):
            source_features = np.load(args.prefix_source + "-feats.npy")
            source_idmap = json.load(open(args.prefix_source+"-id_map.json","r"))
            target_features = np.load(args.prefix_target + "-feats.npy")
            target_idmap = json.load(open(args.prefix_target+"-id_map.json","r"))

        src_ind2word = {v:k for k,v in src_word2ind.items()}
        trg_ind2word = {v:k for k,v in trg_word2ind.items()}


        def neighbor_of_true_nodes(recommend_node, true_nodes):
            for node in true_nodes:
                if trg_ind2word[recommend_node] in G.neighbors(trg_ind2word[node]):
                    return True
            return False

        def same_feature(src_feat, trg_feat):
            return np.sum(src_feat==trg_feat) == len(src_feat)

        result_print = {}
        result_distance = {}
        num_neib_of_true_nodes = 0
        num_same_feats = 0
        for i in src:
            result_print[i] = []
            result_distance[i] = []
            for similar_node in translation[i]:
                # check same features
                str_print = str(trg_ind2word[similar_node])
                # check true node, neighbor of true nodes
                if similar_node in src2trg[i]:
                    str_print += "(true)"
                elif neighbor_of_true_nodes(similar_node, src2trg[i]):
                    str_print += "(neib)"
                    num_neib_of_true_nodes += 1
                else:
                    str_print += "      "

                if source_features is not None:
                    src_node_id = src_ind2word[i]
                    trg_node_id = trg_ind2word[similar_node]
                    if same_feature(source_features[source_idmap[src_node_id]], target_features[target_idmap[trg_node_id]]):
                        str_print += "(feat)"
                        num_same_feats += 1
                    else:
                        str_print += "      "

                result_print[i].append(str_print)

                # compute cosine distance
                distance = 1 - xp.sum(x[i]*z[similar_node])
                result_distance[i].append("{0:.4f}        ".format(distance))

            result_print[i] = ",\t".join(result_print[i])
            result_distance[i] = ",\t".join(result_distance[i])

        # print result
        for i in src:
            print("-----{0}------------------------------".format(src_ind2word[i]))
            print(result_print[i])
            print(result_distance[i])

        k = args.neighborhood
        num_node = len(src)
        total_sample_results = k * num_node
        print("Rate of neighbor of target prediction {0:.4f}".format(num_neib_of_true_nodes/total_sample_results))
        print("Rate of same feature {0:.4f}".format(num_same_feats/total_sample_results))

        matches = np.array([
            1 if len(set(translation[i])) > len(set(translation[i])-set(src2trg[i]))
            else 0 for i in src
        ])


    else:
        for i in src:
            if count % 2000 == 0:
                print("-----{0}-----".format(i))
                print(src2trg[i])
                print(translation[i])
            count += 1
        matches = np.array([1 if translation[i] in src2trg[i] else 0 for i in src])



    accuracy = np.mean(matches) #Every source in the dictionary is forced to have a target (no thresolding)
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))


if __name__ == '__main__':
    main()
