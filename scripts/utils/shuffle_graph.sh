DS=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX=ppi

python data_utils/shuffle_graph.py --input_dir ${DS} --out_dir ${DS}/permutation --prefix ${PREFIX}


