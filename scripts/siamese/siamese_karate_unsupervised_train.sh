OUTDIR=visualize/karate_siamese_clone
MODEL=graphsage_mean
DS=example_data/karate
DS1=example_data/karate
PREFIX1=karate
DS2=example_data/karate
PREFIX2=karate
TRAIN_RATIO=0.8
LR=0.01

############################### compare karate vs karate using vecmap ###############
source activate pytorch

# normal save to original 
python -m graphsage.siamese_unsupervised_train --epochs 10 --model ${MODEL} \
    --prefix_source ${DS1}/graphsage/${PREFIX1} \
    --prefix_target ${DS2}/graphsage/${PREFIX2} \
    --identity_dim 64 \
    --learning_rate ${LR} \
    --save_embeddings True --base_log_dir ${OUTDIR} \
    --train_dict_dir ${DS}/dictionaries/groundtruth \
    --val_dict_dir ${DS}/dictionaries/groundtruth \
    --embedding_loss_weight 1 --mapping_loss_weight 1 \
    --validate_iter 10 \
    --cuda True

source activate vecmap

python vecmap/eval_translation.py ${OUTDIR}/unsup-example_data/graphsage_mean_0.010000/source.emb \
        ${OUTDIR}/unsup-example_data/graphsage_mean_0.010000/target.emb \
        -d ${DS}/dictionaries/groundtruth --retrieval csls --cuda
###################################################################################
