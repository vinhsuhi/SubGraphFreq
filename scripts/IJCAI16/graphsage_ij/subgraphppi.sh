BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
PREFIX2=ppi

nohup python -u  -m IJCAI16.main_graphsage \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.001 \
--learning_rate2 0.005 \
--embedding_dim 150 \
--dim_1 150 \
--embedding_epochs 10 \
--identity_dim 0 \
--mapping_epochs 10 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB3vsSELF \
--train_percent 0.5 \
--batch_size_embedding 512 \
--batch_size_mapping 16 \
--use_features \
--n_layer 2 \
--cuda > graphsageg.txt &




nohup python -u  -m IJCAI16.main_graphsage \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.001 \
--learning_rate2 0.005 \
--embedding_dim 150 \
--dim_1 150 \
--embedding_epochs 100 \
--identity_dim 50 \
--mapping_epochs 100 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB3vsSELF50 \
--train_percent 0.5 \
--batch_size_embedding 512 \
--batch_size_mapping 16 \
--use_features \
--n_layer 2 \
--cuda > graphsageg50.txt &





BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
PREFIX2=ppi

python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name PPISUB3vsSELFg \
--train_percent 0.5 \
--batch_size_embedding 512 \
--batch_size_mapping 16 \
--cuda > ijcaig.txt &



python data_utils/evaluate_distance.py \
--embedding_path $HOME/dataspace/IJCAI16_results/embeddings/s_PPISUB3vsSELFg_elr001_ebsz_edim300ee_500_neg10_trainpercent0.5.npy \
--edgelist_path $HOME/dataspace/graph/ppi/sub_graph/subgraph3/graphsage/ppi_edgelist1.npy \
--idmap_path $HOME/dataspace/graph/ppi/sub_graph/subgraph3/graphsage/ppi-id_map.json \
--file_format "numpy"


OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
PREFIX2=ppi
TRAIN_RATIO=0.5

python -m graphsage.eval_topk \
${OUTDIR}/PPISUB3vsSELFg_elr001_ebsz_edim300ee_500_neg10_trainpercent0.5source.emb \
${OUTDIR}/PPISUB3vsSELFg_elr001_ebsz_edim300ee_500_neg10_trainpercent0.5target.emb \
-d ${DS2}/dictionaries/groundtruth \
-k 5 \
--retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
--prefix_target ${DS2}/graphsage/${PREFIX2}


OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
PREFIX2=ppi
TRAIN_RATIO=0.5

python -m graphsage.eval_topk \
${OUTDIR}/PPISUB3vsSELFg_elr001_ebsz_edim300ee_500_neg10_trainpercent0.5source.emb \
${OUTDIR}/PPISUB3vsSELFg_elr001_ebsz_edim300ee_500_neg10_trainpercent0.5target.emb \
-d ${DS2}/dictionaries/groundtruth \
-k 1 \
--retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
--prefix_target ${DS2}/graphsage/${PREFIX2}




BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate

python -u  -m IJCAI16.main \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.01 \
--learning_rate2 0.005 \
--embedding_dim 300 \
--embedding_epochs 500 \
--mapping_epochs 600 \
--neg_sample_size 2 \
--base_log_dir ${BLD} \
--log_name karatevsself \
--train_percent 0.5 \
--batch_size_embedding 12 \
--batch_size_mapping 5 \
--cuda > karate_ijcaig.txt &

BLD=$HOME/dataspace/IJCAI16_results
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate


nohup python -u  -m IJCAI16.main_graphsage \
--prefix1 ${DS1}/graphsage/${PREFIX1} \
--prefix2 ${DS2}/graphsage/${PREFIX2} \
--ground_truth ${DS2}/dictionaries/groundtruth \
--learning_rate1 0.001 \
--learning_rate2 0.005 \
--embedding_dim 150 \
--dim_1 150 \
--embedding_epochs 100 \
--identity_dim 50 \
--mapping_epochs 100 \
--neg_sample_size 10 \
--base_log_dir ${BLD} \
--log_name karatevsself50 \
--train_percent 0.5 \
--batch_size_embedding 12 \
--batch_size_mapping 5 \
--n_layer 2 \
--cuda > karate_graphsageg50.txt &


karatevsself_elr001_ebsz_edim300ee_500_neg2_trainpercent0.5source.emb
karatevsself50karate_elr0001_ee100_ebsz12_edim300_neg10_GRAPHSAGEsource.emb

OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate
TRAIN_RATIO=0.5

python -m graphsage.eval_topk \
${OUTDIR}/karatevsself_elr001_ebsz_edim300ee_500_neg2_trainpercent0.5source.emb \
${OUTDIR}/karatevsself_elr001_ebsz_edim300ee_500_neg2_trainpercent0.5target.emb \
-d ${DS2}/dictionaries/groundtruth \
-k 1 \
--retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
--prefix_target ${DS2}/graphsage/${PREFIX2}


OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate
TRAIN_RATIO=0.5

python -m graphsage.eval_topk \
${OUTDIR}/karatevsself50karate_elr0001_ee100_ebsz12_edim300_neg10_GRAPHSAGEsource.emb \
${OUTDIR}/karatevsself50karate_elr0001_ee100_ebsz12_edim300_neg10_GRAPHSAGEtarget.emb \
-d ${DS2}/dictionaries/groundtruth \
-k 1 \
--retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
--prefix_target ${DS2}/graphsage/${PREFIX2}





python data_utils/evaluate_distance.py \
--embedding_path $HOME/dataspace/IJCAI16_results/embeddings/s_karatevsself_elr001_ebsz_edim300ee_500_neg2_trainpercent0.5.npy \
--edgelist_path $HOME/dataspace/graph/karate/graphsage/karate_edgelist1.npy \
--idmap_path $HOME/dataspace/graph/karate/graphsage/karate-id_map.json \
--file_format "numpy"

t_karatevsself_elr001_ebsz_edim300ee_500_neg2_trainpercent0.5.npy

s_karatevsself50karate_elr0001_ee100_ebsz12_edim300_neg10_GRAPHSAGE.npy


python data_utils/evaluate_distance.py \
--embedding_path $HOME/dataspace/IJCAI16_results/embeddings/s_karatevsself50karate_elr0001_ee100_ebsz12_edim300_neg10_GRAPHSAGE.npy \
--edgelist_path $HOME/dataspace/graph/karate/graphsage/karate_edgelist1.npy \
--idmap_path $HOME/dataspace/graph/karate/graphsage/karate-id_map.json \
--file_format "numpy"




PPISUB3vsSELFppi_elr0001_ee200_ebsz512_edim300_neg10_GRAPHSAGEsource.emb




OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
PREFIX2=ppi
TRAIN_RATIO=0.5

python -m graphsage.eval_topk \
${OUTDIR}/PPISUB3vsSELFppi_elr0001_ee200_ebsz512_edim300_neg10_GRAPHSAGEsource.emb \
${OUTDIR}/PPISUB3vsSELFppi_elr0001_ee200_ebsz512_edim300_neg10_GRAPHSAGEtarget.emb \
-d ${DS2}/dictionaries/groundtruth \
-k 1 \
--retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
--prefix_target ${DS2}/graphsage/${PREFIX2}



PPISUB3vsSELF50ppi_elr0001_ee100_ebsz512_edim300_neg10_GRAPHSAGEsource.emb




OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
PREFIX2=ppi
TRAIN_RATIO=0.5

python -m graphsage.eval_topk \
${OUTDIR}/PPISUB3vsSELF50ppi_elr0001_ee100_ebsz512_edim300_neg10_GRAPHSAGEsource.emb \
${OUTDIR}/PPISUB3vsSELF50ppi_elr0001_ee100_ebsz512_edim300_neg10_GRAPHSAGEsource.emb \
-d ${DS2}/dictionaries/groundtruth \
-k 1 \
--retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
--prefix_target ${DS2}/graphsage/${PREFIX2}


sam_PPISUB3vsSELFppi_elr0001_ee200_ebsz512_edim300_neg10_GRAPHSAGE.npy



python data_utils/evaluate_distance.py \
--embedding_path $HOME/dataspace/IJCAI16_results/embeddings/sam_PPISUB3vsSELFppi_elr0001_ee200_ebsz512_edim300_neg10_GRAPHSAGE.npy \
--edgelist_path $HOME/dataspace/graph/ppi/sub_graph/subgraph3/edgelist/ppi.edgelist \
--idmap_path $HOME/dataspace/graph/ppi/sub_graph/subgraph3/graphsage/ppi-id_map.json \



python data_utils/evaluate_distance.py \
--embedding_path $HOME/vinh/graphsage_pytorch/graphsage_tf/unsup-graphsage/graphsage_mean_small_0.000010/val.emb.txt \
--edgelist_path $HOME/dataspace/graph/ppi/sub_graph/subgraph3/edgelist/ppi.edgelist \
--idmap_path $HOME/dataspace/graph/ppi/sub_graph/subgraph3/graphsage/ppi-id_map.json \