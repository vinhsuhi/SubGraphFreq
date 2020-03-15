OUTDIR=visualize/douban_offline_online_loadmodel
MODEL=graphsage_mean
DS=example_data/douban
DS1=example_data/douban/offline
PREFIX1=offline
DS2=example_data/douban/online
PREFIX2=online
TRAIN_RATIO=0.8
CLONE_RATIO=0.2



python -m graphsage.unsupervised_train --prefix ${DS1}/graphsage/${PREFIX1} --epochs 20 --model ${MODEL} \
	--save_embeddings True --base_log_dir ${OUTDIR}/original --save_model True --load_walks True \
	--cuda True


python data_utils/random_clone_add.py --input ${DS1}/graphsage --output ${DS1}/random \
    --prefix ${PREFIX1} --padd ${CLONE_RATIO} --nadd ${CLONE_RATIO}



python -m graphsage.unsupervised_train --prefix ${DS1}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX1} --epochs 20 --model ${MODEL} \
	--save_embeddings True --base_log_dir ${OUTDIR}/target --load_model_dir ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/ --load_walks True \
	--cuda True



time python vecmap/map_embeddings.py --semi_supervised ${DS1}/dictionaries/node,split=0.8.train.dict \
    ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
    ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
    --cuda

python vecmap/eval_translation.py ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
	${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
	-d ${DS1}/dictionaries/node,split=0.8.test.dict \
	--retrieval csls --cuda

# identity_dim
python -m graphsage.unsupervised_train --prefix ${DS1}/graphsage/${PREFIX1} --epochs 20 --model ${MODEL} \
	--no_feature True --identity_dim 64 \
	--save_embeddings True --base_log_dir ${OUTDIR}/original --save_model True --load_walks True \
	--cuda True


python data_utils/random_clone_add.py --input ${DS1}/graphsage --output ${DS1}/random \
    --prefix ${PREFIX1} --padd ${CLONE_RATIO} --nadd ${CLONE_RATIO}



python -m graphsage.unsupervised_train --prefix ${DS1}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX1} --epochs 20 --model ${MODEL} \
	--save_embeddings True --base_log_dir ${OUTDIR}/target --load_model_dir ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/ --load_walks True \
	--no_feature True --identity_dim 64 \
	--cuda True




time python vecmap/map_embeddings.py --semi_supervised ${DS1}/dictionaries/node,split=0.8.train.dict \
    ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/all.emb \
    ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
    ${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
    --cuda

python vecmap/eval_translation.py ${OUTDIR}/original/unsup-douban/${MODEL}_0.010000/vecmap.emb \
	${OUTDIR}/target/unsup-douban/${MODEL}_0.010000/vecmap.emb \
	-d ${DS1}/dictionaries/node,split=0.8.test.dict \
	--retrieval csls --cuda
