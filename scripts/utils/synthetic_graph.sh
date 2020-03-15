DS=$HOME/dataspace/graph/synthetic
PREFIX=fully_connect

python data_utils/synthetic_graph.py --prefix ${PREFIX} --min_degree 999 --max_degree 999
