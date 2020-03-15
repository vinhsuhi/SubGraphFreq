RATIO=0.2
python data_utils/random_clone_delete.py --input example_data/douban/offline/graphsage --output example_data/douban/offline/random --prefix offline --pdel 0.2 --ndel 0.2
python data_utils/graphsage_to_mat.py --prefix example_data/douban/offline/graphsage/offline --prefix2 example_data/douban/offline/randomclone,p=0.2,n=0.2/offline --groundtruth example_data/douban/offline/dictionaries/groundtruth.dict --out example_data/douban/offline/mat/douban_offline_delete,p=0.2,n=0.2/offline
