OUTDIR=visualize/ppi_siamese
MODEL=graphsage_mean
DS=$HOME/dataspace/graph/ppi
DS1=$HOME/dataspace/graph/ppi
PREFIX1=ppi
# DS2=example_data/ppi/randomclone,p=0.1,n=0.1
DS2=$HOME/dataspace/graph/ppi
PREFIX2=ppi
TRAIN_RATIO=0.2
LR=0.1
SPLIT=0.2

############################### compare ppi1 and ppi2 using vecmap ###############
source activate pytorch
python data_utils/gen_dict.py --input ${DS1}/graphsage/${PREFIX1}-G.json \
       --dict ${DS1}/dictionaries --split ${SPLIT}

# normal save to original 
python -m graphsage.siamese_supervised_train --epochs 1000 --model ${MODEL} \
    --prefix_source ${DS1}/graphsage/${PREFIX1} \
    --prefix_target ${DS2}/graphsage/${PREFIX2} \
    --learning_rate ${LR} \
    --identity_dim 64 \
    --multiclass True \
	--save_embeddings True --base_log_dir ${OUTDIR} \
    --train_dict_dir ${DS}/dictionaries/node,split=${SPLIT}.train.dict \
    --val_dict_dir ${DS}/dictionaries/node,split=${SPLIT}.test.dict \
    --embedding_loss_weight 1 --mapping_loss_weight 1 \
    --cuda True

source activate vecmap

python vecmap/eval_translation.py ${OUTDIR}/sup/${MODEL}/source.emb \
	${OUTDIR}/sup/${MODEL}/target.emb \
	-d ${DS}/dictionaries/node,split=${SPLIT}.test.dict \
    --neighborhood 1 \
	--retrieval csls 
###################################################################################
