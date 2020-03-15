############################### permute karate using vecmap ###############
OUTDIR=visualize/karate_siamese_permutation
MODEL=graphsage_mean
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate
TRAIN_RATIO=0.5
LR=0.001

# OUTDIR=visualize/ppi_siamese_permutation
# MODEL=graphsage_mean
# DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
# PREFIX1=ppi
# DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
# PREFIX2=ppi
# TRAIN_RATIO=0.5
# LR=0.01

python -m graphsage.eval_topk \
    ${OUTDIR}/unsup/graphsage_mean_0.00/source.emb \
    ${OUTDIR}/unsup/graphsage_mean_0.00/target.emb \
    -d ${DS2}/dictionaries/groundtruth \
    -k 5 \
    --retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
    --prefix_target ${DS2}/graphsage/${PREFIX2}
