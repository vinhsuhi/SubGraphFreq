time python -m graphsage.unsupervised_train --prefix example_data/ppi/graphsage/ppi --epochs 20 --max_degree 25 --model graphsage_mean --cuda True --load_walks True --save_val_embeddings True --base_log_dir visualize/ppi1
# time python -m graphsage.unsupervised_train --prefix example_data/ppi/graphsage/ppi --epochs 20 --max_degree 25 --model graphsage_mean --cuda True --load_walks True --save_val_embeddings True --base_log_dir visualize/ppi2
# python graphsage/visualize.py --embed_dir visualize/ppi1 --out_dir visualize/ppi1 
# python graphsage/visualize.py --embed_dir visualize/ppi2 --out_dir visualize/ppi2
