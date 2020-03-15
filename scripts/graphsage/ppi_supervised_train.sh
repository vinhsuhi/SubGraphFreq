############################### permute ppi ###############
OUTDIR=visualize/ppi
MODEL=graphsage_mean
DS=$HOME/dataspace/graph/ppi
PREFIX=ppi
LR=0.01

############################### compare ppi vs ppi permute using vecmap ###############
source activate pytorch

time python -m graphsage.supervised_train --epochs 100 --model graphsage_sum \
     --prefix ${DS}/graphsage/${PREFIX} \
     --dim_1 128 \
     --dim_2 128 \
     --learning_rate ${LR} \
     --max_degree 25 \
     --cuda True  --save_embeddings True \
     --base_log_dir ${OUTDIR} \
     --multiclass True

