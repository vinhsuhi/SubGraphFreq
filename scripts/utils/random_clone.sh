OUTDIR=visualize/ppi_original_random_loadmodel_0.8
MODEL=graphsage_mean
DS=example_data/ppi
PREFIX=ppi
TRAIN_RATIO=0.8

python data_utils/random_clone_add.py --input example_data/ppi/graphsage/ --output example_data/ppi/random \
    --prefix ppi --padd 0.01 --nadd 0.01

python data_utils/random_clone_add.py --input example_data/ppi/graphsage/ --output example_data/ppi/random \
    --prefix ppi --padd 0.05 --nadd 0.05

python data_utils/random_clone_add.py --input example_data/ppi/graphsage/ --output example_data/ppi/random \
    --prefix ppi --padd 0.1 --nadd 0.1    
