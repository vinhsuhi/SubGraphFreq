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
