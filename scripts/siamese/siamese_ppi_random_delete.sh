OUTDIR=visualize/karate_siamese_random_delete
MODEL=graphsage_mean
PREFIX=ppi
DS=$HOME/dataspace/graph/${PREFIX}
DS1=$HOME/dataspace/graph/${PREFIX}
DS2=$HOME/dataspace/graph/${PREFIX}/random_delete/del,p=
TRAIN_RATIO=0.8
RATIO=0.0
LR=0.01

############################### compare karate vs karate using vecmap ###############
source activate pytorch

python data_utils/random_delete_node.py --input ${DS}/graphsage --output ${DS}/random_delete \
    --prefix ${PREFIX} --ratio ${RATIO}

python data_utils/split_dict.py --input ${DS2}${RATIO}/groundtruth --out_dir ${DS2}${RATIO}/dictionaries/ --split ${TRAIN_RATIO}

# normal save to original 
python -m graphsage.siamese_unsupervised_train --epochs 100 --model ${MODEL} \
    --prefix_source ${DS1}/graphsage/${PREFIX} \
    --prefix_target ${DS2}${RATIO}/${PREFIX} \
    --identity_dim 64 \
    --learning_rate ${LR} \
    --save_embeddings True --base_log_dir ${OUTDIR} \
    --train_dict_dir ${DS2}${RATIO}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    --val_dict_dir ${DS2}${RATIO}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
    --embedding_loss_weight 1 --mapping_loss_weight 1 \
    --validate_iter 1 \
    --cuda True

# source activate vecmap

# python vecmap/eval_translation.py ${OUTDIR}/unsup/graphsage_mean_0.010000/source.emb \
#         ${OUTDIR}/unsup/graphsage_mean_0.010000/target.emb \
#         -d ${DS}/dictionaries/groundtruth --retrieval csls --cuda
# ###################################################################################


