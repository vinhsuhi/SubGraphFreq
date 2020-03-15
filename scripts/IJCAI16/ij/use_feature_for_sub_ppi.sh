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
--log_name PPISUB2vsSELFuseFEATURESbe \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 32 \
--use_features \
--add_feature_before_map \
--cuda > PPISUB2vsSELFuseFEATURESbe02.txt


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
--log_name PPISUB2vsSELFuseFEATURESbe \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 32 \
--use_features \
--add_feature_before_map \
--cuda > PPISUB2vsSELFuseFEATURESbe06.txt









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
--log_name PPISUB2vsSELFuseFEATURESaf \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 32 \
--use_features \
--cuda > PPISUB2vsSELFuseFEATURESaf02.txt



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
--log_name PPISUB2vsSELFuseFEATURESaf \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 32 \
--use_features \
--cuda > PPISUB2vsSELFuseFEATURESaf06.txt








########### add edge 0.1

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
--log_name PPISUB2vsADD01useFEATURESbf \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 16 \
--add_feature_before_map \
--cuda > PPISUB2vsADD01useFEATURESaf06

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
--log_name PPISUB2vsADD01useFEATURESaf \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsADD01useFEATURESaf06


BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph2
PREFIX2=ppi
CLONE_RATIO=0.2




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
--log_name PPISUB2vsADD02useFEATURESaf \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsADD02useFEATURESaf02

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
--log_name PPISUB2vsADD02useFEATURESaf \
--train_percent 0.6 \
--batch_size_embedding 512 \
--batch_size_mapping 16 \
--cuda > PPISUB2vsADD02useFEATURESaf06


