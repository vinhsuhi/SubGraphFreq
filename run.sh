python experiment.py --model GCN --num_adds 500 --data_name mico --large_graph_path GraMi/Datasets/mico.lg --dir data/mico --prefix data/mico/graphsage/mico --epochs 10 --batch_size 10000
cd gSpan

python -m gspan_mining -s 499 -p True ./graphdata/mico_10_3.outx --clabel ./graphdata/mico_10_3.outxatt_label_center


./grami -f mico_10_3.lg -s 14000 -t 0 -p 0


python experiment.py --model GCN --num_adds 500 --data_name mico_10_3 --large_graph_path GraMi/Datasets/mico_10_3.lg --dir data/mico_10_3 --prefix data/mico_10_3/graphsage/mico_10_3 --epochs 10 --batch_size 10000

cd gSpan 


scp -P 15458 vinhtv@0.tcp.ngrok.io:vinh/SubGraphFreq/GraMi/Datasets/mico_10_3.lg .

scp -P 10137 toannt@0.tcp.ngrok.io:/home/toannt/workspace/SubGraphFreq/embeddings.tsv .
scp -P 10137 toannt@0.tcp.ngrok.io:/home/toannt/workspace/SubGraphFreq/k_means_labels.tsv .

python experiment.py --model GCN --num_adds 100 --data_name citeseer --large_graph_path GraMi/Datasets/citeseer.lg --dir data/citeseer --prefix data/citeseer/graphsage/citeseer --epochs 10 --batch_size 500 --output_dim 20
