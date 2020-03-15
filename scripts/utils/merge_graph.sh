############################### permute karate using vecmap ###############
OUTDIR=$HOME/dataspace/graph/karate/merge
MODEL=graphsage_mean
DS1=$HOME/dataspace/graph/karate
PREFIX1=karate
DS2=$HOME/dataspace/graph/karate/permutation
PREFIX2=karate

python -m data_utils.merge_graphs \
    --prefix1 ${DS1}/graphsage/${PREFIX1} \
    --prefix2 ${DS2}/graphsage/${PREFIX2} \
    --out_dir ${OUTDIR} \
    --out_prefix ppi