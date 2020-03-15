python data_utils/merge_graphs.py --prefix1 $HOME/dataspace/graph/ppi/graphsage/ppi --prefix2 \
    $HOME/dataspace/graph/ppi/permutation/graphsage/ppi --out_prefix ppi --out_dir \
    $HOME/dataspace/graph/ppi/merge/ --groundtruth \
    $HOME/dataspace/graph/ppi/permutation/dictionaries/groundtruth

cd graphsage_tf
python -u -m graphsage.unsupervised_train --train_prefix $HOME/dataspace/graph/ppi/merge/graphsage/ppi --epochs \
    10 --save_embeddings True --base_log_dir ../visualize/ppi --model graphsage_mean --print_every 1000 --identity_dim 64 --gpu 1 > ppi &

cd ..
python data_utils/split_embeddings.py --embedding_file visualize/ppi/unsup-graphsage/graphsage_mean_small_0.000010/val.emb.txt --source_ids \
    $HOME/dataspace/graph/ppi/merge/graphsage/ppi-source_ids.npy --target_ids \
    $HOME/dataspace/graph/ppi/merge/graphsage/ppi-target_ids.npy --out_dir \
    visualize/ppi/unsup-graphsage/graphsage_mean_small_0.000010/

python -m graphsage.eval_topk visualize/ppi/unsup-graphsage/graphsage_mean_small_0.000010/source.emb \
    visualize/ppi/unsup-graphsage/graphsage_mean_small_0.000010/target.emb -d \
    $HOME/dataspace/graph/ppi/merge/dictionaries/groundtruth --retrieval csls


# python -m graphsage.siamese_unsupervised_train --prefix_source $HOME/dataspace/graph/ppi/merge/graphsage/ppi --prefix_target \
#     ppi --epochs 1 --map_fc identity --use_random_walks True





# ######### KArate

python data_utils/merge_graphs.py --prefix1 $HOME/dataspace/graph/karate/graphsage/karate --prefix2 \
    $HOME/dataspace/graph/karate/permutation/graphsage/karate --out_prefix karate --out_dir \
    $HOME/dataspace/graph/karate/merge/ --groundtruth \
    $HOME/dataspace/graph/karate/permutation/dictionaries/groundtruth

cd graphsage_tf
python -u -m graphsage.unsupervised_train --train_prefix $HOME/dataspace/graph/karate/merge/graphsage/karate --epochs \
    10 --save_embeddings True --base_log_dir ../visualize/karate --model graphsage_mean --print_every 1000 --identity_dim 64 --gpu 1

cd ..
python data_utils/split_embeddings.py --embedding_file visualize/karate/unsup-graphsage/graphsage_mean_small_0.000010/val.emb.txt --source_ids \
    $HOME/dataspace/graph/karate/merge/graphsage/karate-source_ids.npy --target_ids \
    $HOME/dataspace/graph/karate/merge/graphsage/karate-target_ids.npy --out_dir \
    visualize/karate/unsup-graphsage/graphsage_mean_small_0.000010/

python -m graphsage.eval_topk visualize/karate/unsup-graphsage/graphsage_mean_small_0.000010/source.emb \
    visualize/karate/unsup-graphsage/graphsage_mean_small_0.000010/target.emb -d \
    $HOME/dataspace/graph/karate/merge/dictionaries/groundtruth --retrieval csls

# source activate pytorch
# python -m graphsage.siamese_unsupervised_train --prefix_source $HOME/dataspace/graph/karate/merge/graphsage/karate --prefix_target \
#     karate --epochs 1 --map_fc identity --use_random_walks True
# source deactivate
