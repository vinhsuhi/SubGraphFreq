OUTDIR=visualize/pale_facebook_siamese_clone
MODEL=graphsage_mean
DS=example_data/pale_facebook
DS1=example_data/pale_facebook/random_clone/sourceclone,alpha_c=0.5,alpha_s=0.9
PREFIX1=pale_facebook
DS2=example_data/pale_facebook/random_clone/targetclone,alpha_c=0.5,alpha_s=0.9
PREFIX2=pale_facebook
TRAIN_RATIO=0.8
LR=0.01

############################### compare ppi1 and ppi2 using vecmap ###############
source activate pytorch

# normal save to original 
python -m graphsage.siamese_unsupervised_train --epochs 10 --model ${MODEL} \
    --prefix_source ${DS1}/${PREFIX1} \
    --prefix_target ${DS2}/${PREFIX2} \
    --identity_dim 128 \
    --learning_rate ${LR} \
    --save_embeddings True --base_log_dir ${OUTDIR} \
    --train_dict_dir ${DS}/dictionaries/node,split=0.8.train.dict \
    --val_dict_dir ${DS}/dictionaries/node,split=0.8.test.dict \
    --embedding_loss_weight 1 --mapping_loss_weight 1 \
    --validate_iter 1000 \
    --cuda True

source activate vecmap

python vecmap/eval_translation.py ${OUTDIR}/unsup-pale_facebook/graphsage_mean_0.010000/source.emb \
        ${OUTDIR}/unsup-pale_facebook/graphsage_mean_0.010000/target.emb \
        -d ${DS}/dictionaries/node,split=0.8.full.dict --retrieval csls --cuda
###################################################################################
