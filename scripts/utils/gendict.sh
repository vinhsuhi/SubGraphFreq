DS=example_data/pale_facebook
PREFIX=pale_facebook
# DS=example_data/ppi
# PREFIX=ppi
RATIO=0.8

python data_utils/gen_dict.py --input ${DS}/graphsage/${PREFIX}-G.json --dict ${DS}/dictionaries --split ${RATIO}