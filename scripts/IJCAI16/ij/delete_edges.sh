BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX2=ppi



python data_utils/random_clone_delete.py \
--input ${DS1}/graphsage/ \
--output ${DS1}/randomdel \
--prefix ${PREFIX1} \
--pdel 0.1 \
--ndel 0.1

python data_utils/random_clone_delete.py \
--input ${DS1}/graphsage/ \
--output ${DS1}/randomdel \
--prefix ${PREFIX1} \
--pdel 0.2 \
--ndel 0.2


BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2/
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2/randomdelclone\,p\=0.1\,n\=0.1/
PREFIX2=ppi



python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL01_003.txt


python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.2 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL01_02.txt


python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL01_06.txt


python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL01 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL01_08.txt


BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2/randomdelclone\,p\=0.2\,n\=0.2/
PREFIX2=ppi





python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL02_003.txt


python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.2 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL02_02.txt


python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL02_06.txt


python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsDEL02 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsDEL02_08.txt