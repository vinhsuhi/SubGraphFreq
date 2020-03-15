BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/random_delete_node/del,p=
PREFIX2=ppi
LR=0.01

# python data_utils/random_delete_node.py --input ${DS1}/graphsage --output ${DS1}/random_delete_node \
#     --prefix ${PREFIX1} --ratio ${RATIO}

# python data_utils/shuffle_graph.py --input_dir ${DS2}${RATIO} --out_dir ${DS2}${RATIO} --prefix ${PREFIX2}

source activate pytorch
RATIO=0.2
TRAIN_RATIO=0.03

python data_utils/split_dict.py --input ${DS2}${RATIO}/dictionaries/groundtruth --out_dir ${DS2}${RATIO}/dictionaries/ --split ${TRAIN_RATIO}

python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}${RATIO}/graphsage/${PREFIX2} \
--train_dict ${DS2}${RATIO}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
--val_dict ${DS2}${RATIO}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
--learning_rate1 ${LR} \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 2000 \
--mapping_epochs 2000 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name pale_ppi_subgraph_del_node_${RATIO}_${TRAIN_RATIO} \
--train_percent 0.03 \
--batch_size_embedding 512 \
--batch_size_mapping 8 \
--cuda > pale_ppi_subgraph_del_node_${RATIO}_${TRAIN_RATIO}.txt


RATIO=0.1
TRAIN_RATIO=0.03

python data_utils/split_dict.py --input ${DS2}${RATIO}/dictionaries/groundtruth --out_dir ${DS2}${RATIO}/dictionaries/ --split ${TRAIN_RATIO}

python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}${RATIO}/graphsage/${PREFIX2} \
--train_dict ${DS2}${RATIO}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
--val_dict ${DS2}${RATIO}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
--learning_rate1 ${LR} \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 2000 \
--mapping_epochs 2000 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name pale_ppi_subgraph_del_node_${RATIO}_${TRAIN_RATIO} \
--train_percent 0.03 \
--batch_size_embedding 512 \
--batch_size_mapping 8 \
--cuda > pale_ppi_subgraph_del_node_${RATIO}_${TRAIN_RATIO}.txt

