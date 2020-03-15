DATASPACE=$HOME/dataspace/graph
#flickr_myspace
# DATASET1=flickr
# DATASET2=myspace
# DS=${DATASPACE}/${DATASET1}_${DATASET2}
# model=small

# flickr_lastfm
# DATASET1=flickr
# DATASET2=lastfm
# DS=${DATASPACE}/${DATASET1}_${DATASET2}
# model=small

# douban
DATASET1=online
DATASET2=offline
PREFIX=douban
DS=${DATASPACE}/${PREFIX}

# source activate emb-tensorflow
time python data_utils/mat_to_graphsage.py \
    --input ${DS}/data.mat --dataset1 ${DATASET1} --dataset2 ${DATASET2} --output ${DS}

