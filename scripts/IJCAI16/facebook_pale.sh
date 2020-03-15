BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/pale_facebook/random_clone/sourceclone\,alpha_c\=0.9\,alpha_s\=0.5
PREFIX1=pale_facebook
DS2=$HOME/dataspace/graph/pale_facebook/random_clone/targetclone\,alpha_c\=0.9\,alpha_s\=0.5
PREFIX2=pale_facebook


python -m IJCAI16.main \
--prefix1 ${DS1}/${PREFIX1} \
--prefix2 ${DS2}/${PREFIX2} \
--ground_truth $HOME/dataspace/graph/pale_facebook/dictionaries/node,split=0.8.full.dict \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 1500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name c09s05 \
--train_percent 0.2 \
--batch_size_embedding 512 \
--batch_size_mapping 64 \
--cuda
