DS=$HOME/dataspace/graph/ppi
PREFIX=ppi

python data_utils/graphsage_to_edgelist.py --input_dir ${DS} --out_dir ${DS} --prefix ${PREFIX}
