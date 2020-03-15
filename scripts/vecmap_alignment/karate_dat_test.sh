DATA=karate
SPLIT=0.2
source activate pytorch
python data_utils/gen_dict.py --input example_data/${DATA}/graphsage/${DATA}-G.json --dict example_data/${DATA}/dictionaries --split ${SPLIT}
python -m graphsage.siamese_unsupervised_train --cuda True --validate_iter 100 --print_every \
    10 --embedding_loss_weight 1 --mapping_loss_weight 2 --prefix_source example_data/${DATA}/graphsage/${DATA} --prefix_target \
    example_data/${DATA}/graphsage/${DATA} --epochs 30 --identity_dim 64 --train_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.train.dict --val_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.test.dict --base_log_dir visualize/${DATA}/

source activate vecmap
python vecmap/eval_translation.py visualize/${DATA}/unsup-example_data/graphsage_mean_0.010000/source.emb \
        visualize/${DATA}/unsup-example_data/graphsage_mean_0.010000/target.emb \
        -d example_data/${DATA}/dictionaries/node,split=${SPLIT}.test.dict --retrieval csls --cuda

