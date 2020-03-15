############################### permute ppi ###############
OUTDIR1=visualize/ppi_siamese_permutation/original_tf
OUTDIR2=visualize/ppi_siamese_permutation/target_tf
MODEL=graphsage_mean
DS1=$HOME/dataspace/graph/ppi/sub_graph/subgraph3
PREFIX1=ppi
DS2=$HOME/dataspace/graph/ppi/sub_graph/subgraph3/permutation
PREFIX2=ppi
TRAIN_RATIO=0.5
LR=0.01

# python data_utils/shuffle_graph.py --input_dir ${DS1} --out_dir ${DS2} --prefix ${PREFIX2}
# python data_utils/split_dict.py --input ${DS2}/dictionaries/groundtruth --out_dir ${DS2}${RATIO}/dictionaries/ --split ${TRAIN_RATIO}

############################### compare ppi vs ppi permute using vecmap ###############
#source activate tensorflow

cd graphsage_tf

# normal save to original 
time python -m graphsage.unsupervised_train --epochs 100 --model graphsage_mean \
     --train_prefix ${DS1}/graphsage/${PREFIX1} \
     --dim_1 128 \
     --dim_2 128 \
     --learning_rate ${LR} \
     --max_degree 25 \
     --gpu 0 --save_embeddings True \
     --base_log_dir ${OUTDIR1} --neg_sample_size 20 \
#    --max_total_steps 1

# time python -m graphsage.unsupervised_train --epochs 1 --model graphsage_mean \
#      --train_prefix ${DS2}/graphsage/${PREFIX2} \
#      --dim_1 128 \
#      --dim_2 128 \
#      --learning_rate ${LR} \
#      --max_degree 25 \
#      --gpu 0 --save_embeddings True \
#      --base_log_dir ${OUTDIR2} \
#      --max_total_steps 1

# python -m graphsage.eval_topk \
#     ${OUTDIR1}/unsup/graphsage_mean_0.010000/all.emb \
#     ${OUTDIR2}/unsup/graphsage_mean_0.010000/all.emb \
#     -d ${DS2}/dictionaries/groundtruth \
#     -k 1 \
#     --retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
#     --prefix_target ${DS2}/graphsage/${PREFIX2}

# cd ..
# python -m graphsage.eval_topk \
#     ${OUTDIR1}/unsup-graphsage/graphsage_mean_small_0.010000/val.emb.txt \
#     ${OUTDIR2}/unsup-graphsage/graphsage_mean_small_0.010000/val.emb.txt \
#     -d ${DS2}/dictionaries/groundtruth \
#     -k 1 \
#     --retrieval topk --prefix_source ${DS1}/graphsage/${PREFIX1} \
#     --prefix_target ${DS2}/graphsage/${PREFIX2}

###################################################################################

# time python -m graphsage.unsupervised_train --prefix example_data/ppi/graphsage/ppi --epochs 20 --max_degree 25 --model graphsage_mean --cuda True --load_walks True --save_val_embeddings True --base_log_dir visualize/ppi2
# python graphsage/visualize.py --embed_dir visualize/ppi1 --out_dir visualize/ppi1 
# python graphsage/visualize.py --embed_dir visualize/ppi2 --out_dir visualize/ppi2
