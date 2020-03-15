OUTDIR=visualize/ppi_original_maxpool_0.8
MODEL=graphsage_mean
DS=example_data/ppi
PREFIX=ppi
TRAIN_RATIO=0.8

source activate pytorch

# normal save to original 
python -m graphsage.supervised_train --prefix ${DS}/graphsage/${PREFIX} --epochs 100 --model ${MODEL} \
	--save_model True --save_embeddings True --multiclass True --base_log_dir ${OUTDIR}/original \
	--save_embedding_samples True --cuda True

# normal save to target 
python -m graphsage.supervised_train --prefix ${DS}/graphsage/${PREFIX} --epochs 100 --model ${MODEL} \
	--save_model True --save_embeddings True --multiclass True --base_log_dir ${OUTDIR}/target \
	--save_embedding_samples True --cuda True

############################### compare ppi1 and ppi2 using vecmap ###############
source activate vecmap

time python vecmap/map_embeddings.py --semi_supervised ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    ${OUTDIR}/original/sup-example_data/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/target/sup-example_data/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/original/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    ${OUTDIR}/target/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    --cuda

python vecmap/eval_translation.py ${OUTDIR}/original/sup-example_data/${MODEL}_0.010000/vecmap.emb \
	${OUTDIR}/target/sup-example_data/${MODEL}_0.010000/vecmap.emb \
	-d ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
	--retrieval csls --cuda
###################################################################################
