DS=example_data/ppi/graphsage/ppi
OUTDIR=visualize

source activate pytorch
time python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --multiclass True --epochs 100 --max_degree 25 --model graphsage_mean --cuda True --save_embeddings True --base_log_dir visualize/ppi1 --save_embeddings True
time python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --multiclass True --epochs 100 --max_degree 25 --model graphsage_mean --cuda True --save_embeddings True --base_log_dir visualize/ppi2 --save_embeddings True 

source activate vecmap
time python vecmap/map_embeddings.py --unsupervised \
    visualize/ppi1/sup-example_data/graphsage_mean_0.010000/all.emb \
    visualize/ppi2/sup-example_data/graphsage_mean_0.010000/all.emb \
    visualize/ppi1/sup-example_data/graphsage_mean_0.010000/vecmap.emb visualize/ppi2/sup-example_data/graphsage_mean_0.010000/vecmap.emb \
    --cuda
python vecmap/eval_translation.py visualize/ppi1/sup-example_data/graphsage_mean_0.010000/vecmap.emb visualize/ppi2/sup-example_data/graphsage_mean_0.010000/vecmap.emb -d example_data/ppi/dictionary/groundtruth.dict --retrieval csls




