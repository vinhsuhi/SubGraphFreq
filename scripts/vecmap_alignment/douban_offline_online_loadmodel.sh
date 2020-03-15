OUTDIR=visualize/douban_offline_online_loadmodel
MODEL=graphsage_mean
DS=example_data/douban
DS1=example_data/douban/offline
PREFIX1=offline
DS2=example_data/douban/online
PREFIX2=online
TRAIN_RATIO=0.8
CLONE_RATIO=0.01

############################### compare ppi1 and ppi2 using vecmap ###############
source activate pytorch

# normal save to original 
# python -m graphsage.unsupervised_train --prefix ${DS1}/graphsage/${PREFIX1} --epochs 20 --model ${MODEL} \
# 	--save_embeddings True --base_log_dir ${OUTDIR}/original --save_model True --load_walks True \
# 	--cuda True

# normal save to target
python -m graphsage.unsupervised_train --prefix ${DS2}/graphsage/${PREFIX2} --epochs 20 --model ${MODEL} \
	--save_embeddings True --base_log_dir ${OUTDIR}/target --load_model_dir ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/ --load_walks True \
	--cuda True

# source activate vecmap

# time python vecmap/map_embeddings.py --semi_supervised ${DS}/dictionaries/groundtruth.dict \
#     ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/all.emb \
#     ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/all.emb \
#     ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
#     ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
#     --cuda

# python vecmap/eval_translation.py ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
# 	${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
# 	-d ${DS}/dictionaries/groundtruth.dict \
# 	--retrieval csls --cuda
###################################################################################
