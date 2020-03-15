DATA=karate
LOG=karate003
SPLIT=0.03
#python data_utils/gen_dict.py --input $HOME/dataspace/graph/${DATA}/graphsage/${DATA}-G.json --dict example_data/${DATA}/dictionaries --split ${SPLIT}
python -m graphsage.siamese_unsup_ind_train --cuda True --print_every \
    50 --prefix_source $HOME/dataspace/graph/${DATA}/graphsage/${DATA} --prefix_target \
    $HOME/dataspace/graph/${DATA}/graphsage/${DATA} --embed_epochs 100 --mapping_epochs 100 --train_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.train.dict --val_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.test.dict --base_log_dir visualize/${DATA}/ --identity_dim 64
