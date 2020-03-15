DS=$HOME/dataspace/graph/karate
PREFIX=karate
RATIO=0.1

python data_utils/random_delete_node.py --input ${DS}/graphsage --output ${DS}/random_delete \
    --prefix ${PREFIX} --ratio ${RATIO}