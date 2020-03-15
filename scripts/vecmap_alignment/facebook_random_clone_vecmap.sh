DATASPACE=example_data/facebook/graphsage
PREFIX=facebook
DS=${DATASPACE}/${PREFIX}
p=0.01
n=0.01

source activate emb-tensorflow
python data_utils/random_clone.py --input ${DS}/graphsage/ --output ${DS}/random_clone/ --prefix ${PREFIX} --padd ${p} --pdel ${p} --nadd ${n} --ndel ${n}
python data_utils/gen_dict.py --input ${DS}/graphsage/${PREFIX}-G.json --dict ${DS}/dictionaries --split 0.2
python data_utils/gen_dict.py --input ${DS}/graphsage/${PREFIX}-G.json --dict ${DS}/dictionaries --split 0.8

# # GraphSAGE
# cd graphsage/
# python -m graphsage.utils ${DS}/graphsage/${PREFIX}-G.json ${DS}/graphsage/${PREFIX}-walks.txt
# time python -m graphsage.unsupervised_train --train_prefix ${DS}/graphsage/${PREFIX} --model graphsage_meanpool --model_size ${model} \
#     --epochs 10 --dropout 0.1 --weight_decay 0.01 --max_total_steps 1000 --validate_iter 10 --gpu 0 --print_every 10\
#     --identity_dim 128 --base_log_dir ${DS}/embeddings

# python -m graphsage.utils ${DS}/random_clone/clone,p=${p},n=${n}/${PREFIX}-G.json ${DS}/random_clone/clone,p=${p},n=${n}/${PREFIX}-walks.txt
# time python -m graphsage.unsupervised_train --train_prefix ${DS}/random_clone/clone,p=${p},n=${n}/${PREFIX} --model graphsage_meanpool --model_size ${model} \
#     --epochs 10 --dropout 0.1 --weight_decay 0.01 --max_total_steps 1000 --validate_iter 10 --gpu 0 --print_every 10\
#      --identity_dim 128 --base_log_dir ${DS}/embeddings/

# cd ../
# source activate vecmap

# time python vecmap/map_embeddings.py --unsupervised \
#     ${DS}/embeddings/unsup-graphsage/gcn_${model}_0.000010/val.emb.txt \
#     ${DS}/embeddings/unsup-clone,p=${p},n=${n}/gcn_${model}_0.000010/val.emb.txt \
#     ${DS}/embeddings/unsup-clone,p=${p},n=${n}/sval_mapped.emb \
#     ${DS}/embeddings/unsup-clone,p=${p},n=${n}/rval_mapped.emb
#     --cuda

# # Evaluate
# python vecmap/eval_translation.py ${DS}/embeddings/unsup-clone,p=${p},n=${n}/sval_mapped.emb ${DS}/embeddings/unsup-clone,p=${p},n=${n}/rval_mapped.emb -d ${DS}/dictionaries/node,split=0.2.test.dict --retrieval nn
# python vecmap/eval_similarity.py -l --backoff 0 ${DS}/embeddings/unsup-clone,p=${p},n=${n}/sval_mapped,0.2.emb ${DS}/embeddings/unsup-clone,p=${p},n=${n}/rval_mapped,0.2.emb -i ${DS}/dictionaries/TEST_SIMILARITY.TXT

# # time python vecmap/map_embeddings.py --semi_supervised ${DS}/dictionaries/node,split=0.8.train.dict \
# #     ${DS}/embeddings/unsup-graphsage/graphsage_maxpool_${model}_0.000010/val.emb.txt \
# #     ${DS}/embeddings/unsup-clone,p=${p},n=${n}/graphsage_maxpool_${model}_0.000010/val.emb.txt \
# #     ${DS}/embeddings/unsup-clone,p=${p},n=${n}/sval_mapped,0.8.emb ${DS}/embeddings/unsup-clone,p=${p},n=${n}/rval_mapped,0.8.emb \
# #     # --cuda
# # python vecmap/eval_translation.py ${DS}/embeddings/unsup-clone,p=${p},n=${n}/sval_mapped,0.8.emb ${DS}/embeddings/unsup-clone,p=${p},n=${n}/rval_mapped,0.8.emb -d ${DS}/dictionaries/node,split=0.8.test.dict --retrieval csls
