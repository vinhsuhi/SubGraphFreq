BLD=$HOME/dataspace/IJCAI16_results
DS=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/random_clone/add_edge,p=0.1,n=0.1
TRAIN_RATIO=0.8
EMB_PREFIX=ppi_subgraph_add_edge_${TRAIN_RATIO}ppi_ee2000_GRAPHSAGE
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
   
