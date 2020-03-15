echo "Test Simultaneously"
DATA=karate
SPLIT=0.2
source activate pytorch
python data_utils/gen_dict.py --input example_data/${DATA}/graphsage/${DATA}-G.json --dict example_data/${DATA}/dictionaries --split ${SPLIT}
python -m graphsage.siamese_unsupervised_train --cuda True --validate_iter 100 --print_every \
    50 --embedding_loss_weight 1 --mapping_loss_weight 2 --prefix_source example_data/${DATA}/graphsage/${DATA} --prefix_target \
    example_data/${DATA}/graphsage/${DATA} --epochs 100 --train_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.train.dict --val_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.test.dict --groundtruth \
    example_data/${DATA}/dictionaries/groundtruth.dict --base_log_dir visualize/${DATA}/ --identity_dim 64
echo "==========================="


echo "Test Simultaneously"
DATA=ppi
SPLIT=0.8
TARGET=example_data/${DATA}/randomaddclone,p=0.1,n=0.1/${DATA}
LOG="ppiadd18.txt"
# source activate pytorch
# python data_utils/gen_dict.py --input example_data/${DATA}/graphsage/${DATA}-G.json --dict example_data/${DATA}/dictionaries --split ${SPLIT}
# python data_utils/random_clone_add.py --input example_data/${DATA}/graphsage/ --output example_data/${DATA}/randomadd --prefix ${DATA} --padd 0.1 --nadd 0.1
# python data_utils/random_clone_delete.py --input example_data/${DATA}/graphsage/ --output example_data/${DATA}/randomdel --prefix ${DATA} --pdel 0.1 --ndel 0.1
python -u -m graphsage.siamese_unsupervised_train --cuda True --validate_iter 5000 --print_every \
    5000 --embedding_loss_weight 1 --mapping_loss_weight 2 --prefix_source example_data/${DATA}/graphsage/${DATA} --prefix_target \
    ${TARGET} --epochs 100 --train_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.train.dict --val_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.test.dict --groundtruth \
    example_data/${DATA}/dictionaries/groundtruth.dict --base_log_dir visualize/${DATA}/ > ${LOG} &
echo "=============================="




echo "Test Independently"
DATA=karate
SPLIT=0.8
python data_utils/gen_dict.py --input example_data/${DATA}/graphsage/${DATA}-G.json --dict example_data/${DATA}/dictionaries --split ${SPLIT}
python -u -m graphsage.siamese_unsup_ind_train --cuda True --validate_iter 100 --print_every \
    500 --prefix_source example_data/${DATA}/graphsage/${DATA} --prefix_target \
    example_data/${DATA}/graphsage/${DATA} --embed_epochs 100 --mapping_epochs 100 --train_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.train.dict --val_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.test.dict --groundtruth \
    example_data/${DATA}/dictionaries/groundtruth.dict --base_log_dir visualize/${DATA}/ --identity_dim 64



echo "Test Independently"
DATA=ppi
SPLIT=0.2
TARGET=example_data/${DATA}/randomaddclone,p=0.1,n=0.1/${DATA}
LOG="ppiindadd12.txt"
python -u -m graphsage.siamese_unsup_ind_train --cuda True --validate_iter 5000 --print_every \
    5000 --prefix_source example_data/${DATA}/graphsage/${DATA} --prefix_target \
    ${TARGET} --embed_epochs 100 --mapping_epochs 100 --train_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.train.dict --val_dict_dir \
    example_data/${DATA}/dictionaries/node,split=${SPLIT}.test.dict --groundtruth \
    example_data/${DATA}/dictionaries/groundtruth.dict --base_log_dir visualize/${DATA}/ > ${LOG} &
echo "=============================="
