BLD=$HOME/dataspace/IJCAI16_results
DS=$HOME/dataspace/graph/douban
DS1=$HOME/dataspace/graph/douban/online
PREFIX1=online
DS2=$HOME/dataspace/graph/douban/offline
PREFIX2=offline
RATIO=0.1
LR=0.01
LOGNAME=douban
OUTDIM=300
EMBEDDINGEPOCHS=2000
MAPPINGEPOCHS=5
NEGSAMPLESIZE=10
BATCHSIZEEMBEDDING=512
BATCHSIZEMAPPING=10


TRAIN_RATIO=0.2

python data_utils/split_dict.py --input ${DS}/dictionaries/groundtruth --out_dir ${DS}/dictionaries/ --split ${TRAIN_RATIO}



python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--train_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
--val_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
--learning_rate1 ${LR} \
--learning_rate2 ${LR} \
--embedding_dim ${OUTDIM} \
--embedding_epochs ${EMBEDDINGEPOCHS} \
--mapping_epochs ${MAPPINGEPOCHS} \
--neg_sample_size ${NEGSAMPLESIZE} \
--base_log_dir ${BLD} \
--log_name ${LOGNAME} \
--train_percent ${TRAIN_RATIO} \
--batch_size_embedding ${BATCHSIZEEMBEDDING} \
--batch_size_mapping ${BATCHSIZEMAPPING} \
--cuda 


# python -u -m Improve_IJCAI.new_model \
#  --prefix1 ${DS1}/graphsage/${PREFIX1} \
#  --prefix2 ${DS2}/graphsage/${PREFIX2} \
# --train_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
# --val_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --learning_rate ${LR} \
# --embedding_dim ${OUTDIM} \
# --num_epochs ${NUMEPOCHS} \
# --neg_sample_size ${NEGSAMPLESIZE} \
# --batch_size_embedding ${BATCHSIZEEMBEDDING} \
# --base_log_dir ${BLD} \
# --log_name ${LOGNAME} \
# --train_percent ${TRAIN_RATIO} \
# --mapping_sample_size ${BATCHSIZEMAPPING} \
# --use_features \
# --cuda 
