OUTDIR=visualize/karate
MODEL=graphsage_mean
DS=$HOME/dataspace/graph/karate
PREFIX=karate

python graphsage/visualize.py --embed_file ${OUTDIR}/unsup/graphsage_mean_0.01/all.npy \
        --out_file ${OUTDIR}/unsup/graphsage_mean_0.01/all.tsv \
        --idmap_path ${DS}/graphsage/${PREFIX}-id_map.json \
        --file_format numpy

# OUTDIR=$HOME/dataspace/IJCAI16_results/embeddings
# MODEL=graphsage_mean
# DS=$HOME/dataspace/graph/karate
# PREFIX=karate

# python graphsage/visualize.py --embed_file ${OUTDIR}/source.npy \
#         --out_file ${OUTDIR}/source.tsv \
#         --idmap_path ${DS}/graphsage/${PREFIX}-id_map.json \
#         --file_format numpy
