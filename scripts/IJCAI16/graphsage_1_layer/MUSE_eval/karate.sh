BLD=$HOME/dataspace/IJCAI16_results
EMB_PREFIX=karate_0.2karate_ee2000_GRAPHSAGE
SOURCE_EMB=${BLD}/embeddings/${EMB_PREFIX}source.emb
TARGET_EMB=${BLD}/embeddings/${EMB_PREFIX}target.emb
DS=$HOME/dataspace/graph/karate/permutation
TRAIN_RATIO=0.2

source activate pytorch 

python MUSE/supervised.py \
    --src_emb ${SOURCE_EMB} \
    --tgt_emb ${TARGET_EMB} \
    --emb_dim 200 \
    --n_refinement 5 \
    --dico_train ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    --dico_eval ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
    --dico_max_rank 0
    
