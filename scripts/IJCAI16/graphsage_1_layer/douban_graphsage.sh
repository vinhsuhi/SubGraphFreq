# DONE


BLD=$HOME/dataspace/IJCAI16_results
DS=$HOME/dataspace/graph/douban
DS1=$HOME/dataspace/graph/douban/online
PREFIX1=online
DS2=$HOME/dataspace/graph/douban/offline
PREFIX2=offline
LR=0.01


source activate pytorch


TRAIN_RATIO=0.8

python data_utils/split_dict.py --input ${DS}/dictionaries/groundtruth --out_dir ${DS}/dictionaries/ --split ${TRAIN_RATIO}

time python -u -m IJCAI16.main_graphsage \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}/graphsage/${PREFIX2} \
 --train_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 0.01 \
 --out_dim 150 \
 --embedding_epochs 2000 \
 --use_features \
 --identity_dim 50 \
 --mapping_epochs 2000 \
 --neg_sample_size 10 \
 --base_log_dir ${BLD} \
 --log_name douban \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding 512 \
 --batch_size_mapping 32 \
 --n_layer 1 \
 --cuda > logs/graphsage_douban_${TRAIN_RATIO}







TRAIN_RATIO=0.2

python data_utils/split_dict.py --input ${DS}/dictionaries/groundtruth --out_dir ${DS}/dictionaries/ --split ${TRAIN_RATIO}

time python -u  -m IJCAI16.main_graphsage \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}/graphsage/${PREFIX2} \
 --train_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 0.01 \
 --out_dim 150 \
 --embedding_epochs 2000 \
 --identity_dim 50 \
 --use_features \
 --mapping_epochs 2000 \
 --neg_sample_size 10 \
 --base_log_dir ${BLD} \
 --log_name douban \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding 512 \
 --batch_size_mapping 16 \
 --n_layer 1 \
 --cuda > logs/graphsage_douban_${TRAIN_RATIO}







TRAIN_RATIO=0.03

python data_utils/split_dict.py --input ${DS}/dictionaries/groundtruth --out_dir ${DS}/dictionaries/ --split ${TRAIN_RATIO}

time python -u  -m IJCAI16.main_graphsage \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}/graphsage/${PREFIX2} \
 --train_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 0.001 \
 --out_dim 150 \
 --embedding_epochs 2000 \
 --use_features \
 --identity_dim 50 \
 --mapping_epochs 100 \
 --neg_sample_size 10 \
 --base_log_dir ${BLD} \
 --log_name douban \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding 512 \
 --batch_size_mapping 8 \
 --n_layer 1 \
 --cuda > logs/graphsage_douban_${TRAIN_RATIO}



