############################### permute ppi using vecmap ###############
OUTDIR=visualize/synthetic_random_10
MODEL=graphsage_mean
DS1=$HOME/dataspace/graph/synthetic/random_10
PREFIX1=random_10
DS2=$HOME/dataspace/graph/synthetic/random_10/permutation
PREFIX2=random_10
TRAIN_RATIO=0.5
LR=0.01

python data_utils/synthetic_graph.py --prefix ${PREFIX1} --visualize True --min_degree 1 --max_degree 29 --num_nodes 30 --feature_dim 10 --out_dir ${DS1}
python data_utils/shuffle_graph.py --input_dir ${DS1} --out_dir ${DS2} --prefix ${PREFIX2}
python data_utils/split_dict.py --input ${DS2}/dictionaries/groundtruth --out_dir ${DS2}${RATIO}/dictionaries/ --split ${TRAIN_RATIO}

############################### compare karate vs karate using vecmap ###############
source activate pytorch

# normal save to original 
python -m graphsage.siamese_unsupervised_train --epochs 100 --model ${MODEL} \
    --prefix_source ${DS1}/graphsage/${PREFIX1} \
    --prefix_target ${DS2}/graphsage/${PREFIX2} \
    --batch_size 10 \
    --dim_1 5 \
    --dim_2 10 \
    --samples_2 0 \
    --learning_rate ${LR} \
    --save_embeddings True --base_log_dir ${OUTDIR} \
    --groundtruth ${DS2}/dictionaries/groundtruth \
    --train_dict_dir ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    --val_dict_dir ${DS2}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
    --embedding_loss_weight 1 --mapping_loss_weight 1 \
    --validate_iter 1000 --neg_sample_size 5 --map_fc identity \
    --cuda True


###################################################################################
