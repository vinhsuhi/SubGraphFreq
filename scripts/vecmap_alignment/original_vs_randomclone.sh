OUTDIR=visualize/ppi_original_random_loadmodel_0.8
MODEL=graphsage_mean
DS=example_data/ppi
PREFIX=ppi
TRAIN_RATIO=0.8

source activate pytorch

# normal 									save to ppi1 
python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --epochs 100 \
	--save_model True --save_embeddings True --multiclass True --base_log_dir visualize/ppi1 \
	--save_embedding_samples True --cuda True

# random clone 0.05
python data_utils/random_clone_add.py --input $HOME/dataspace/graph/ppi/graphsage/ --output $HOME/dataspace/graph/ppi/random \
    --prefix ppi --padd 0.1 --nadd 0.1

# old model + random clone
python -m graphsage.supervised_train --epochs 100 \
	--load_model_dir visualize/ppi1/sup-example_data/graphsage_mean_0.010000/ \
	--save_embeddings True --multiclass True --base_log_dir visualize/ppi2  \
	--prefix example_data/ppi/randomclone\,p\=0.1\,n\=0.1/ppi --cuda True


############################### compare ppi1 and ppi2 using vecmap ###############
source activate vecmap

time python vecmap/map_embeddings.py --unsupervised \
    /home/paperspace/graphsage_pytorch/visualize/ppi1/sup-example_data/graphsage_mean_0.010000/all.emb \
    /home/paperspace/graphsage_pytorch/visualize/ppi2/sup-example_data/graphsage_mean_0.010000/all.emb \
    /home/paperspace/graphsage_pytorch/visualize/ppi1/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
    /home/paperspace/graphsage_pytorch/visualize/ppi2/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
    --cuda

python vecmap/eval_translation.py /home/paperspace/graphsage_pytorch/visualize/ppi1/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
	/home/paperspace/graphsage_pytorch/visualize/ppi2/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
	-d /home/paperspace/graphsage_pytorch/example_data/ppi/dictionaries/groundtruth.dict \
	--retrieval csls --cuda
###################################################################################


time python vecmap/map_embeddings.py --semi_supervised ${DS}/dictionaries/node,split=${TRAIN_RATIO}.train.dict \
    /home/paperspace/graphsage_pytorch/visualize/ppi1/sup-example_data/graphsage_mean_0.010000/all.emb \
    /home/paperspace/graphsage_pytorch/visualize/ppi2/sup-example_data/graphsage_mean_0.010000/all.emb \
    /home/paperspace/graphsage_pytorch/visualize/ppi1/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
    /home/paperspace/graphsage_pytorch/visualize/ppi2/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
    --cuda

python vecmap/eval_translation.py /home/paperspace/graphsage_pytorch/visualize/ppi1/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
	/home/paperspace/graphsage_pytorch/visualize/ppi2/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
	-d ${DS}/dictionaries/node,split=${TRAIN_RATIO}.test.dict \
	--retrieval csls --cuda