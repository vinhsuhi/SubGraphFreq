OUTDIR=visualize/karate_siamese_permutation
MODEL=graphsage_mean
DS=$HOME/dataspace/graph/karate
PREFIX=karate
TRAIN_RATIO=0.5
LR=0.001

# OUTDIR=visualize/ppi_siamese_permutation
# MODEL=graphsage_mean
# DS=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
# PREFIX=ppi
# TRAIN_RATIO=0.5
# LR=0.01

source activate pytorch

python data_utils/evaluate_distance.py \
    --embedding_path ${OUTDIR}/unsup/graphsage_mean_0.00/source.emb \
    --edgelist_path ${DS}/edgelist/${PREFIX}.edgelist \
    --idmap_path ${DS}/graphsage/${PREFIX}-id_map.json

# OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
# MODEL=graphsage_mean
# DS=$HOME/dataspace/graph/karate
# PREFIX=karate

# python data_utils/evaluate_distance.py \
#     --embedding_path ${OUTDIR}/target.npy \
#     --edgelist_path ${DS}/edgelist/${PREFIX}.edgelist \
#     --idmap_path ${DS}/graphsage/${PREFIX}-id_map.json \
#     --file_format numpy

