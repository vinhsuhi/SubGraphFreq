BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/flickr_myspace
PREFIX1=flickr
DS2=$HOME/dataspace/graph/flickr_myspace
PREFIX2=myspace
LR=0.01
LOGNAME=flickr_myspace
OUTDIM=300
EMBEDDINGEPOCHS=2000
NEGSAMPLESIZE=10
BATCHSIZEEMBEDDING=512



TRAIN_RATIO=0.2
MAPPINGEPOCHS=2000
BATCHSIZEMAPPING=32


python -u -m IJCAI16.main \
 --prefix1 ${DS1}/${PREFIX1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}/${PREFIX2}/graphsage/${PREFIX2} \
 --train_dict ${DS1}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 ${LR} \
 --embedding_dim ${OUTDIM} \
 --embedding_epochs ${EMBEDDINGEPOCHS} \
 --mapping_epochs ${MAPPINGEPOCHS} \
 --neg_sample_size ${NEGSAMPLESIZE} \
 --base_log_dir ${BLD} \
 --log_name ${LOGNAME}_train_percent${TRAIN_RATIO} \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding ${BATCHSIZEEMBEDDING} \
 --batch_size_mapping ${BATCHSIZEMAPPING} \
 --cuda