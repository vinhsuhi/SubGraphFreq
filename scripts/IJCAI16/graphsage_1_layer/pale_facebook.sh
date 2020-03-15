#DONE

source activate pytorch


DS=$HOME/dataspace/graph/pale_facebook
PREFIX1=pale_facebook
PREFIX2=pale_facebook
ALS=0.9
ALC=0.9
DS1=${DS}/random_clone/sourceclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
DS2=${DS}/random_clone/targetclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
TRAIN_RATIO=0.2
LR=0.01
BLD=$HOME/dataspace/IJCAI16_results
EMBEDDINGEPOCHS=500
MAPPINGEPOCHS=500
NEGSAMPLESIZE=10
OUTDIM=10
IDENTITYDIM=150
LOGNAME=pale0909
BATCHSIZEEMBEDDING=512
BATCHSIZEMAPPING=128


python data_utils/pale_random_clone.py \
--input ${DS}/graphsage \
--output1 ${DS}/random_clone \
--output2 ${DS}/random_clone \
--alpha_c ${ALC} --alpha_s ${ALS} --prefix ${PREFIX1}


python data_utils/shuffle_graph.py \
--input_dir ${DS2} \
--out_dir ${DS2} \
--prefix ${PREFIX1}


python data_utils/split_dict.py \
--input ${DS2}/dictionaries/groundtruth \
--out_dir ${DS2}/dictionaries \
--split ${TRAIN_RATIO}


python -u  -m IJCAI16.main_graphsage \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--train_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
--val_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
--learning_rate1 ${LR} \
--learning_rate2 0.001 \
--out_dim ${OUTDIM} \
--embedding_epochs ${EMBEDDINGEPOCHS} \
--identity_dim ${IDENTITYDIM} \
--mapping_epochs ${MAPPINGEPOCHS} \
--neg_sample_size ${NEGSAMPLESIZE} \
--base_log_dir ${BLD} \
--log_name pale0909 \
--train_percent ${TRAIN_RATIO} \
--batch_size_embedding ${BATCHSIZEEMBEDDING} \
--batch_size_mapping ${BATCHSIZEMAPPING} \
--cuda > logs/${LOGNAME}_${TRAIN_RATIO} 



# TRAIN_RATIO=0.03
# LR=0.01
# BLD=$HOME/dataspace/IJCAI16_results

# python data_utils/pale_random_clone.py \
# --input ${DS}/graphsage \
# --output1 ${DS}/random_clone \
# --output2 ${DS}/random_clone \
# --alpha_c ${ALC} --alpha_s ${ALS} --prefix ${PREFIX1}


# python data_utils/shuffle_graph.py \
# --input_dir ${DS2} \
# --out_dir ${DS2} \
# --prefix ${PREFIX1}


# python data_utils/split_dict.py \
# --input ${DS2}/dictionaries/groundtruth \
# --out_dir ${DS2}/dictionaries \
# --split ${TRAIN_RATIO}


# python -u  -m IJCAI16.main_graphsage \
# --prefix1 ${DS1}/graphsage/${PREFIX1} \
# --prefix2 ${DS2}/graphsage/${PREFIX2} \
# --train_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --val_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --learning_rate1 ${LR} \
# --learning_rate2 0.001 \
# --out_dim 150 \
# --embedding_epochs 500 \
# --identity_dim 150 \
# --mapping_epochs 500 \
# --neg_sample_size 10 \
# --base_log_dir ${BLD} \
# --log_name pale0909 \
# --train_percent ${TRAIN_RATIO} \
# --batch_size_embedding 512 \
# --batch_size_mapping 128 \
# --n_layer 1 \
# --cuda > pale_facebook0909_${TRAIN_RATIO} &


# DS=$HOME/dataspace/graph/pale_facebook
# PREFIX1=pale_facebook
# PREFIX2=pale_facebook
# ALS=0.5
# ALC=0.9
# DS1=${DS}/random_clone/sourceclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
# DS2=${DS}/random_clone/targetclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
# TRAIN_RATIO=0.2
# LR=0.01
# BLD=$HOME/dataspace/IJCAI16_results

# python data_utils/pale_random_clone.py \
# --input ${DS}/graphsage \
# --output1 ${DS}/random_clone \
# --output2 ${DS}/random_clone \
# --alpha_c ${ALC} --alpha_s ${ALS} --prefix ${PREFIX1}


# python data_utils/shuffle_graph.py \
# --input_dir ${DS2} \
# --out_dir ${DS2} \
# --prefix ${PREFIX1}


# python data_utils/split_dict.py \
# --input ${DS2}/dictionaries/groundtruth \
# --out_dir ${DS2}/dictionaries \
# --split ${TRAIN_RATIO}


# python -u  -m IJCAI16.main_graphsage \
# --prefix1 ${DS1}/graphsage/${PREFIX1} \
# --prefix2 ${DS2}/graphsage/${PREFIX2} \
# --train_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --val_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --learning_rate1 ${LR} \
# --learning_rate2 0.001 \
# --out_dim 150 \
# --embedding_epochs 500 \
# --identity_dim 150 \
# --mapping_epochs 500 \
# --neg_sample_size 10 \
# --base_log_dir ${BLD} \
# --log_name pale0509 \
# --train_percent ${TRAIN_RATIO} \
# --batch_size_embedding 512 \
# --batch_size_mapping 64 \
# --n_layer 1 \
# --cuda > logs/pale_facebook0509_${TRAIN_RATIO} &



# DS=$HOME/dataspace/graph/pale_facebook
# PREFIX1=pale_facebook
# PREFIX2=pale_facebook
# ALS=0.6
# ALC=0.5
# DS1=${DS}/random_clone/sourceclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
# DS2=${DS}/random_clone/targetclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
# TRAIN_RATIO=0.2
# LR=0.01
# BLD=$HOME/dataspace/IJCAI16_results

# python data_utils/pale_random_clone.py \
# --input ${DS}/graphsage \
# --output1 ${DS}/random_clone \
# --output2 ${DS}/random_clone \
# --alpha_c ${ALC} --alpha_s ${ALS} --prefix ${PREFIX1}


# python data_utils/shuffle_graph.py \
# --input_dir ${DS2} \
# --out_dir ${DS2} \
# --prefix ${PREFIX1}


# python data_utils/split_dict.py \
# --input ${DS2}/dictionaries/groundtruth \
# --out_dir ${DS2}/dictionaries \
# --split ${TRAIN_RATIO}


# python -u  -m IJCAI16.main_graphsage \
# --prefix1 ${DS1}/graphsage/${PREFIX1} \
# --prefix2 ${DS2}/graphsage/${PREFIX2} \
# --train_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --val_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --learning_rate1 ${LR} \
# --learning_rate2 0.001 \
# --out_dim 150 \
# --embedding_epochs 500 \
# --identity_dim 150 \
# --mapping_epochs 500 \
# --neg_sample_size 10 \
# --base_log_dir ${BLD} \
# --log_name pale0605 \
# --train_percent ${TRAIN_RATIO} \
# --batch_size_embedding 512 \
# --batch_size_mapping 64 \
# --n_layer 1 \
# --cuda > pale_facebook0605_${TRAIN_RATIO} 



# DS=$HOME/dataspace/graph/pale_facebook
# PREFIX1=pale_facebook
# PREFIX2=pale_facebook
# ALS=0.6
# ALC=0.5
# DS1=${DS}/random_clone/sourceclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
# DS2=${DS}/random_clone/targetclone\,alpha_c\=${ALC}\,alpha_s\=${ALS}
# TRAIN_RATIO=0.2
# LR=0.01
# BLD=$HOME/dataspace/IJCAI16_results

# python data_utils/pale_random_clone.py \
# --input ${DS}/graphsage \
# --output1 ${DS}/random_clone \
# --output2 ${DS}/random_clone \
# --alpha_c ${ALC} --alpha_s ${ALS} --prefix ${PREFIX1}


# python data_utils/shuffle_graph.py \
# --input_dir ${DS2} \
# --out_dir ${DS2} \
# --prefix ${PREFIX1}


# python data_utils/split_dict.py \
# --input ${DS2}/dictionaries/groundtruth \
# --out_dir ${DS2}/dictionaries \
# --split ${TRAIN_RATIO}


# python -u  -m IJCAI16.main_graphsage \
# --prefix1 ${DS1}/graphsage/${PREFIX1} \
# --prefix2 ${DS2}/graphsage/${PREFIX2} \
# --train_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --val_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
# --learning_rate1 ${LR} \
# --learning_rate2 0.001 \
# --out_dim 150 \
# --embedding_epochs 500 \
# --identity_dim 150 \
# --mapping_epochs 500 \
# --neg_sample_size 10 \
# --base_log_dir ${BLD} \
# --log_name pale0605 \
# --train_percent ${TRAIN_RATIO} \
# --batch_size_embedding 512 \
# --batch_size_mapping 64 \
# --n_layer 1 \
# --cuda > pale_facebook0605_${TRAIN_RATIO}

