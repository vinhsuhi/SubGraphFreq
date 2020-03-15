# Test Pytorch
source activate pytorch
python data_utils/random_clone.py --input example_data/facebook/graphsage/ --output \
    example_data/facebook/random --prefix facebook --padd 0.01 --pdel 0.01 --nadd 0.01 --ndel 0.01
python -m graphsage.unsupervised_train --prefix example_data/facebook/graphsage/facebook --multiclass \
    True --epochs 20 --max_degree 25 --model graphsage_mean --cuda True --save_val_embeddings True --base_log_dir \
    visualize/facebook/py1 --load_walks True --print_every 200 --identity_dim 64
python -m graphsage.unsupervised_train --prefix example_data/facebook/randomclone,p=0.01,n=0.01/facebook --multiclass \
    True --epochs 20 --max_degree 25 --model graphsage_mean --cuda True --save_val_embeddings True --base_log_dir \
    visualize/facebook/py2 --load_walks True --print_every 200 --identity_dim 64

source activate vecmap
time python vecmap/map_embeddings.py --unsupervised \
    visualize/facebook/py1/unsup-example_data/graphsage_mean_0.010000/all.emb \
    visualize/facebook/py2/unsup-example_data/graphsage_mean_0.010000/all.emb \
    visualize/facebook/py1/unsup-example_data/graphsage_mean_0.010000/vecmap.emb \
    visualize/facebook/py2/unsup-example_data/graphsage_mean_0.010000/vecmap.emb \
    --cuda
python vecmap/eval_translation.py visualize/facebook/py1/unsup-example_data/graphsage_mean_0.010000/vecmap.emb \
    visualize/facebook/py2/unsup-example_data/graphsage_mean_0.010000/vecmap.emb -d \
    example_data/facebook/dictionary/groundtruth.dict --retrieval csls


# Test TF
source activate tensorflow
cd graphsage_tf
python -m graphsage.unsupervised_train --train_prefix ../example_data/facebook/graphsage/facebook --epochs \
    20 --max_degree 25 --model graphsage_mean --save_embeddings True --base_log_dir \
    ../visualize/facebook/tf1 --print_every 200 --identity_dim 64 --gpu 0
python -m graphsage.unsupervised_train --train_prefix ../example_data/facebook/randomclone,p=0.01,n=0.01/facebook --epochs \
    20 --max_degree 25 --model graphsage_mean --save_embeddings True --base_log_dir \
    ../visualize/facebook/tf2 --print_every 200 --identity_dim 64 --gpu 0
cd ..

source activate vecmap
time python vecmap/map_embeddings.py --unsupervised \
    visualize/facebook/tf1/unsup-graphsage/graphsage_mean_small_0.010000/val.emb.txt \
    visualize/facebook/tf2/unsup-randomclone,p=0.01,n=0.01/graphsage_mean_small_0.010000/val.emb.txt \
    visualize/facebook/tf1/unsup-graphsage/graphsage_mean_small_0.010000/vecmap.emb \
    visualize/facebook/tf2/unsup-randomclone,p=0.01,n=0.01/graphsage_mean_small_0.010000/vecmap.emb \
    --cuda
python vecmap/eval_translation.py visualize/facebook/tf1/unsup-graphsage/graphsage_mean_small_0.010000/vecmap.emb \
    visualize/facebook/tf2/unsup-randomclone,p=0.01,n=0.01/graphsage_mean_small_0.010000/vecmap.emb -d \
    example_data/facebook/dictionary/groundtruth.dict --retrieval csls