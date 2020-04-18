python experiment.py --model Graphsage --num_adds 150 --data_name mico --large_graph_path data/mico.edges

cd gSpan

python -m gspan_mining -s 15 -p True ./graphdata/file2.outx -l 