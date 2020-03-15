# pale_facebook: s05 c09 trainpc 0.03 | s09 c09 trainpc 0.2 (1500 epochs | batch_size 600)


BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/pale_facebook/random_clone/sourceclone\,alpha_c\=0.9\,alpha_s\=0.5
DS2=$HOME/dataspace/graph/pale_facebook/random_clone/targetclone\,alpha_c\=0.9\,alpha_s\=0.5
PREFIX=pale_facebook

# train 0.03

python -u -m IJCAI16.main \
--prefix1 ${DS1}/${PREFIX} \
--prefix2 ${DS2}/${PREFIX} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 1500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PALE_FACEBOOK_s05c09 \
--train_percent 0.03 \
--batch_size_embedding 256 \
--batch_size_mapping 256 \
--embedding_dir1 /home/vinhsuhi/dataspace/IJCAI16_results/embeddings/source_pale.npy \
--embedding_dir2 /home/vinhsuhi/dataspace/IJCAI16_results/embeddings/target_pale.npy  \
--cuda > PALE_FACEBOOK_s05c09_003


BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/pale_facebook/random_clone/sourceclone\,alpha_c\=0.9\,alpha_s\=0.9
DS2=$HOME/dataspace/graph/pale_facebook/random_clone/targetclone\,alpha_c\=0.9\,alpha_s\=0.9
PREFIX=pale_facebook
# train 0.2

python -u -m IJCAI16.main \
--prefix1 ${DS1}/${PREFIX} \
--prefix2 ${DS2}/${PREFIX} \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 1500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PALE_FACEBOOK_s09c09 \
--train_percent 0.2 \
--batch_size_embedding 256 \
--batch_size_mapping 16 \
--cuda > PALE_FACEBOOK_s09c09_02




