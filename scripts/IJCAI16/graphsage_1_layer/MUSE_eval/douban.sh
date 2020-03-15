BLD=$HOME/dataspace/IJCAI16_results
DS=$HOME/dataspace/graph/douban
TRAIN_RATIO=0.8
EMB_PREFIX=douban_${TRAIN_RATIO}online_ee2000_GRAPHSAGE
SOURCE_EMB=${BLD}/embeddings/${EMB_PREFIX}source.emb
TARGET_EMB=${BLD}/embeddings/${EMB_PREFIX}target.emb

source activate pytorch 

python MUSE/supervised.py \
    --src_emb ${SOURCE_EMB} \
    --tgt_emb ${TARGET_EMB} \
    --emb_dim 300 \
    --n_refinement 5 \
    --dico_train ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    --dico_eval ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
    --dico_max_rank 0
    


