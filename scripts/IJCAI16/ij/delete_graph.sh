################
# lr1: 0.01, lr2: 0.005   
# pale_facebook: s05 c09 trainpc 0.03 | s09 c09 trainpc 0.2 (1500 epochs | batch_size 600)
# ppi_subgraph2: self train pc 0.03, 0.2, 0.6, 0.8 (500 epcochs| mapping epcochs 600), add 0.1, add 0.2, del 0.1, del 0.2
# ppi_delete01, ppi_delete02: train pc 0.2 (1500 epochs | batch_size 600)
################


#### small dataset first

# ppisubgraph self train 0.03

BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX2=ppi

python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsSELF \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda


# ppisubgraph self train 0.2

python -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsSELF \
--train_percent 0.2 \
--batch_size_embedding 256 \
--batch_size_mapping 128 \
--cuda

# ppisubgraph self train 0.6

python -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsSELF \
--train_percent 0.6 \
--batch_size_embedding 256 \
--batch_size_mapping 256 \
--cuda

# ppisubgraph self train 0.8

python -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsSELF \
--train_percent 0.8 \
--batch_size_embedding 256 \
--batch_size_mapping 256 \
--cuda



## ppisubgraph add 01



BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX2=ppi
CLONE_RATIO=0.1

python data_utils/random_clone_add.py --input ${DS1}/graphsage --output ${DS1}/random \
    --prefix ${PREFIX1} --padd 0.1 --nadd 0.1

python data_utils/random_clone_add.py --input ${DS1}/graphsage --output ${DS1}/random \
    --prefix ${PREFIX1} --padd 0.2 --nadd 0.2

# train 0.03


python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD01 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda




# train 0.03


python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD01 \
--train_percent 0.2 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda

# train 0.6


python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD01 \
--train_percent 0.6 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda

# train 0.8


python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD01 \
--train_percent 0.8 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda



## ppisubgraph add 02



BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX2=ppi
CLONE_RATIO=0.2

# train 0.03

python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD02 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda




# train 0.03


python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD02 \
--train_percent 0.2 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda

# train 0.6


python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD02 \
--train_percent 0.6 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda

# train 0.8


python -u -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/randomclone\,p\=${CLONE_RATIO}\,n\=${CLONE_RATIO}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsADD02 \
--train_percent 0.8 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda





# delete 0.1

BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
PREFIX2=ppi
CLONE_RATIO=0.1
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2/rdelete/del\,p\=${CLONE_RATIO}/permutation


python data_utils/random_delete_node.py  \
    --input ${DS1}/graphsage \
    --output ${DS1}/rdelete \
    --ratio 0.1 \
    --prefix ppi


python data_utils/random_delete_node.py  \
    --input ${DS1}/graphsage \
    --output ${DS1}/rdelete \
    --ratio 0.2 \
    --prefix ppi


python data_utils/random_delete_node.py  \
    --input ${DS1}/graphsage \
    --output ${DS1}/rdelete \
    --ratio 0.4 \
    --prefix ppi


python data_utils/shuffle_graph.py --input_dir ${DS1}/rdelete/del\,p\=${CLONE_RATIO} --out_dir ${DS2} --prefix ${PREFIX2}

# train 0.03


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.03 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda


# train 0.2


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda

# train 0.6


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda


# train 0.8


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.8 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda



# delete 0.2


BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
PREFIX2=ppi
CLONE_RATIO=0.2
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2/rdelete/del\,p\=${CLONE_RATIO}/permutation


python data_utils/shuffle_graph.py --input_dir ${DS1}/rdelete/del\,p\=${CLONE_RATIO} --out_dir ${DS2} --prefix ${PREFIX2}

# train 0.03


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.03 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda


# train 0.2


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda

# train 0.6


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda


# train 0.8


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.8 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda



# delete 0.4



BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
PREFIX2=ppi
CLONE_RATIO=0.4
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2/rdelete/del\,p\=${CLONE_RATIO}/permutation


python data_utils/shuffle_graph.py --input_dir ${DS1}/rdelete/del\,p\=${CLONE_RATIO} --out_dir ${DS2} --prefix ${PREFIX2}

# train 0.03


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL04 \
--train_percent 0.03 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda


# train 0.2


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL04 \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda

# train 0.6


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL04 \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda


# train 0.8


python -m IJCAI16.main \
--prefix1 ${DS0}/graphsage/${PREFIX0} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.01 \
--embedding_dim 300 \
--embedding_epochs 200 \
--mapping_epochs 200 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL04 \
--train_percent 0.8 \
--batch_size_embedding 512 \
--batch_size_mapping 128 \
--cuda
