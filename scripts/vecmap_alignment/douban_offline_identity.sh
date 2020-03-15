OUTDIR=visualize/douban_offline_no_feature
MODEL=graphsage_mean
DS=example_data/douban/offline
PREFIX=offline
TRAIN_RATIO=0.8
CLONE_RATIO=0.01

############################### compare ppi1 and ppi2 using vecmap ###############
source activate pytorch

# normal save to original 
python -m graphsage.unsupervised_train --prefix ${DS}/graphsage/${PREFIX} --epochs 20 --model ${MODEL} \
	--save_embeddings True --base_log_dir ${OUTDIR}/original --load_walks TRUE --no_feature True --identity_dim 64 \
	--cuda True

normal save to original 
python -m graphsage.unsupervised_train --prefix ${DS}/graphsage/${PREFIX} --epochs 20 --model ${MODEL} \
	--save_embeddings True --base_log_dir ${OUTDIR}/target --load_walks TRUE --no_feature True --identity_dim 64 \
	--cuda True

source activate vecmap

time python vecmap/map_embeddings.py --unsupervised \
    ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
    ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
    --cuda

python vecmap/eval_translation.py ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
	${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
	-d ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
	--retrieval csls --cuda
###################################################################################
