python experiment.py --model GCN --num_adds 150 --data_name mico --large_graph_path data/mico.edges --dir data/mico --prefix data/mico/graphsage/mico

cd gSpan

python -m gspan_mining -s 15 -p True ./graphdata/file2.outx -l 