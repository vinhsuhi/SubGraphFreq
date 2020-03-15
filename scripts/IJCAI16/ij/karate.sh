############################### permute ppi using vecmap ###############
OUTDIR=visualize/karate_siamese_permutation
MODEL=graphsage_mean
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate
TRAIN_RATIO=0.8
LR=0.01

python data_utils/shuffle_graph.py --input_dir ${DS1} --out_dir ${DS2} --prefix ${PREFIX2}
python data_utils/split_dict.py --input ${DS2}/dictionaries/groundtruth --out_dir ${DS2}${RATIO}/dictionaries/ --split ${TRAIN_RATIO}

python -m IJCAI16.main \
    --prefix1 ${DS1}/graphsage/${PREFIX1} \
    --prefix2 ${DS2}/graphsage/${PREFIX2} \
    --ground_truth ${DS2}/dictionaries/groundtruth \
    --learning_rate1 0.01 \
    --learning_rate2 0.01 \
    --embedding_dim 300 \
    --embedding_epochs 100 \
    --mapping_epochs 300 \
    --neg_sample_size 10 \
    --base_log_dir $HOME/dataspace/IJCAI16_results \
    --log_name ppivsppi \
    --train_percent 0.8 \
    --batch_size_embedding 16 \
    --batch_size_mapping 8 \
    --cuda 