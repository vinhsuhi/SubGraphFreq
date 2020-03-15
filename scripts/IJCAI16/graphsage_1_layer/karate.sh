############################### permute karate using vecmap ###############
BLD=$HOME/dataspace/IJCAI16_results
OUTDIR=visualize/karate_siamese_permutation
MODEL=graphsage_mean
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate
TRAIN_RATIO=0.2
LR=0.01

# python data_utils/shuffle_graph.py --input_dir ${DS1} --out_dir ${DS2} --prefix ${PREFIX2}

############################### compare karate vs karate using vecmap ###############
source activate pytorch
python data_utils/split_dict.py --input ${DS2}/dictionaries/groundtruth --out_dir ${DS2}/dictionaries/ --split ${TRAIN_RATIO}

time python -u -m IJCAI16.main_graphsage \
 --prefix1 ${DS1}/graphsage/${PREFIX1} \
 --prefix2 ${DS2}/graphsage/${PREFIX2} \
 --train_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
 --val_dict ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
 --learning_rate1 ${LR} \
 --learning_rate2 0.01 \
 --dim_2 100 \
 --dim_1 100 \
 --embedding_epochs 2000 \
 --identity_dim 100 \
 --mapping_epochs 2000 \
 --neg_sample_size 2 \
 --base_log_dir ${BLD} \
 --log_name karate_${TRAIN_RATIO} \
 --train_percent ${TRAIN_RATIO} \
 --batch_size_embedding 8 \
 --batch_size_mapping 2 \
 --n_layer 1 \
 --cuda 

