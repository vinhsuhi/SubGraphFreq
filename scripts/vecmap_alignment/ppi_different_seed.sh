DS=example_data/ppi/graphsage/ppi
OUTDIR=visualize
MODEL=graphsage_maxpool

# Pytorch
source activate pytorch
time python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --multiclass True --epochs 100 --max_degree 25 --model ${MODEL} --cuda True \
     --save_embeddings True --save_embedding_samples True --base_log_dir visualize/ppi_different_seed/original
time python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --multiclass True --epochs 100 --max_degree 25 --model ${MODEL} --cuda True \
     --save_embeddings True --base_log_dir visualize/ppi_different_seed/seed_100 --seed 100\
     --load_embedding_samples_dir visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/ \
     --load_adj_dir visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000

source activate vecmap
time python vecmap/map_embeddings.py --unsupervised \
    visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/all.emb \
    visualize/ppi_different_seed/seed_100/sup-example_data/${MODEL}_0.010000/all.emb \
    visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    visualize/ppi_different_seed/seed_100/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    --cuda
python vecmap/eval_translation.py visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    visualize/ppi_different_seed/seed_100/sup-example_data/${MODEL}_0.010000/vecmap.emb -d example_data/ppi/dictionary/groundtruth.dict --retrieval csls


# TF
source activate tensorflow
cd graphsage_tf
time python -m graphsage.supervised_train --train_prefix ../example_data/ppi/graphsage/ppi --epochs 100 --max_degree 25 --model ${MODEL} \
     --save_embeddings True --save_embedding_samples True --base_log_dir ../visualize/ppi_different_seed/original --gpu 0
time python -m graphsage.supervised_train --train_prefix ../example_data/ppi/graphsage/ppi --epochs 100 --max_degree 25 --model ${MODEL} \
     --save_embeddings True --base_log_dir ../visualize/ppi_different_seed/seed_100 --seed 100\
     --load_embedding_samples_dir ../visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/ \
     --load_adj_dir ../visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000 --gpu 0
cd ..
source activate vecmap
time python vecmap/map_embeddings.py --unsupervised \
    visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/all.emb \
    visualize/ppi_different_seed/seed_100/sup-example_data/${MODEL}_0.010000/all.emb \
    visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    visualize/ppi_different_seed/seed_100/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    --cuda
python vecmap/eval_translation.py visualize/ppi_different_seed/original/sup-example_data/${MODEL}_0.010000/vecmap.emb \
    visualize/ppi_different_seed/seed_100/sup-example_data/${MODEL}_0.010000/vecmap.emb -d example_data/ppi/dictionary/groundtruth.dict --retrieval csls


SOURCE=visualize/IJCAI16/source/
TARGET=visualize/IJCAI16/target/
SOURCEM=visualize/IJCAI16/source_with_map/


time python vecmap/map_embeddings.py --unsupervised \
    ${SOURCE}all.emb \
    ${TARGET}all.emb \
    ${SOURCE}vecmap.emb \
    ${TARGET}vecmap.emb \
    --cuda  

time python vecmap/eval_translation.py ${SOURCE}/vecmap.emb \
    ${TARGET}/vecmap.emb -d example_data/ppi/dictionary/groundtruth.dict --retrieval csls