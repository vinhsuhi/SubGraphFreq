DS=$HOME/dataspace/graph/douban
PREFIX1=online
PREFIX2=offline

python data_utils/feature_groundtruth_checking.py --dict_file ${DS}/dictionaries/groundtruth \
       --source_dataset_dir ${DS}/${PREFIX1}/graphsage \
       --target_dataset_dir ${DS}/${PREFIX2}/graphsage \
       --source_dataset_prefix ${PREFIX1} \
       --target_dataset_prefix ${PREFIX2} 