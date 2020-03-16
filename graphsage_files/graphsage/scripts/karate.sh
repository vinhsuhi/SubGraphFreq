time python -m graphsage.unsupervised_train --prefix example_data/karate/graphsage/karate --batch_size 64 --epochs 100 --max_degree 25 --model graphsage_mean --identity_dim 64 --load_walks True --save_val_embeddings True --cuda True
python graphsage/visualize.py --embed_dir unsup-example_data/graphsage_mean_0.010000 --out_dir visualize/karate_graphsage_mean_0.010000 
