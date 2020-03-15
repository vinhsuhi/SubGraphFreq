python data_utils/mat_to_graphsage.py \
--input $HOME/dataspace/graph/flickr_lastfm/data.mat \
--dataset1 flickr \
--dataset2 lastfm \
--output $HOME/dataspace/graph/flickr_lastfm



python data_utils/mat_to_graphsage.py \
--input $HOME/dataspace/graph/flickr_myspace/data.mat \
--dataset1 flickr \
--dataset2 myspace \
--output $HOME/dataspace/graph/flickr_myspace


python data_utils/mat_to_graphsage.py \
--input $HOME/dataspace/graph/douban/data.mat \
--dataset1 offline \
--dataset2 online \
--output $HOME/dataspace/graph/douban



BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/flickr_lastfm
PREFIX1=flickr
DS2=$HOME/dataspace/graph/flickr_lastfm
PREFIX2=lastfm

# flickr_lastfm
python -u  -m IJCAI16.main \
--prefix1 ${DS1}/${PREFIX1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2}/graphsage/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsSELF \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 10 \
--ground_truth ${DS2}/dictionaries/groundtruth \
--cuda > flickr_lastfm.txt



BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/flickr_myspace
PREFIX1=flickr
DS2=$HOME/dataspace/graph/flickr_myspace
PREFIX2=myspace

# flickr_myspace
python -u  -m IJCAI16.main \
--prefix1 ${DS1}/${PREFIX1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2}/graphsage/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsSELF \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 10 \
--ground_truth ${DS2}/dictionaries/groundtruth \
--cuda > flickr_lastfm.txt





BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/douban
PREFIX1=online
DS2=$HOME/dataspace/graph/douban
PREFIX2=offline

python -u  -m IJCAI16.main \
--prefix1 ${DS1}/${PREFIX1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2}/graphsage/${PREFIX2} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB2vsSELF \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 50 \
--ground_truth ${DS2}/dictionaries/groundtruth \
--cuda > douban.txt


