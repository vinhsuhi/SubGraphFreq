
# compare trained model and loaded model
source activate pytorch

# normal 									save to ppi1 
python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --epochs 20 \
	--save_model True --save_embeddings True --multiclass True --base_log_dir visualize/ppi1 \
	--save_embedding_samples True --cuda True

# load all old. 							save to ppi2 and load from ppi1
python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --epochs 20 \
	--load_model_dir visualize/ppi1/sup-example_data/graphsage_mean_0.010000/ --save_embeddings True \
	--multiclass True --base_log_dir visualize/ppi2 \
	--load_adj_dir visualize/ppi1/sup-example_data/graphsage_mean_0.010000/ \
	--load_embedding_samples_dir visualize/ppi1/sup-example_data/graphsage_mean_0.010000/ --cuda True



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
	-d /home/paperspace/graphsage_pytorch/example_data/ppi/dictionary/groundtruth.dict \
	--retrieval csls --cuda
###################################################################################
