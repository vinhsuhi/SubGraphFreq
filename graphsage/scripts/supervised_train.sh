time python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --multiclass True --epochs 1 --max_degree 25 --model graphsage_maxpool --cuda True --save_embeddings True --base_log_dir visualize/ppi_maxpool
#python graphsage/visualize.py --embed_dir sup-example_data/graphsage_mean_0.010000 --out_dir visualize/ppi_sup_graphsage_mean_0.010000 
