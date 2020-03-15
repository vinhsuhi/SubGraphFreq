OUTDIR=visualize/karate
MODEL=graphsage_mean
PREFIX=karate
DS=$HOME/dataspace/graph/${PREFIX}
TRAIN_RATIO=0.8
RATIO=0.1
LR=0.001

############################### compare karate vs karate using vecmap ###############
source activate pytorch

# normal save to original 
python -m graphsage.unsupervised_train --epochs 1000 --model ${MODEL} \
    --prefix ${DS}/graphsage/${PREFIX} \
    --batch_size 8 --print_every 10 \
    --identity_dim 128 \
    --samples_1 8 --samples_2 4 \
    --learning_rate ${LR} \
    --save_embeddings True --base_log_dir ${OUTDIR} \
    --neg_sample_size 2\
    --cuda True \
    --dim_1 128 --dim_2 128 --save_model True

for i in 0.1 0.2 0.3
do
	python -u BootEA.py ../../data/dbp15k/zh_en/0_3/ ${i} > output/zhen_del_edges_30_${i}
done

for i in 0.1 0.2 0.3
do
	python -u BootEA.py ../../data/dbp15k/ja_en/0_3/ ${i} > output/jaen_del_edges_30_${i}
done

for i in 0.1 0.2 0.3
do
	python -u BootEA.py ../../data/dbp15k/fr_en/0_3/ ${i} > output/fren_del_edges_30_${i}
done

##
for i in 0.1 0.2 0.3
do
	python -u main.py --language zh_en --del_node ${i} > output/zhen_del_edges_${i}
done

for i in 0.1 0.2 0.3
do
	python -u main.py --language ja_en --del_node ${i} > output/jaen_del_edges_${i}
done

for i in 0.1 0.2 0.3
do
	python -u main.py --language fr_en --del_node ${i} > output/fren_del_edges_${i}
done
 
##
cd ../GCN-Align/
for i in 0.1 0.2 0.3
do
	python -u train.py --epochs 2000 --lang zh_en --del_edges ${i} > output/zh_en_del_edges_${i}
done

for i in 0.1 0.2 0.3
do
	python -u train.py --epochs 2000 --lang ja_en --del_edges ${i} > output/ja_en_del_edges_${i}
done

for i in 0.1 0.2 0.3
do
	python -u train.py --epochs 2000 --lang fr_en --del_edges ${i} > output/fr_en_del_edges_${i}
done

# jape
for j in zh_en ja_en fr_en
do
	for i in 1 2 3 4 5
	do
		python attr2vec.py ../../data/dbp15k/${j}/ ../../data/dbp15k/${j}/0_3/del_atts_0_${i}/ ../../data/dbp15k/${j}/all_attrs_range ../../data/dbp15k/en_all_attrs_range
		python ent2vec_sparse.py ../../data/dbp15k/${j}/ 0.3/del_atts_0_${i} 0.95 0.95 0.9
		python -u cse_pos_neg.py ../../data/dbp15k/${j}/ 0.3/del_atts_0_${i} > output1/${j}_delatts_0_${i}
	done
done

for i in 1 2 3
do
	python attr2vec.py ../../data/dbp15k/ja_en/ ../../data/dbp15k/ja_en/0_3/del_atts_0_${i}/ ../../data/dbp15k/ja_en/all_attrs_range ../../data/dbp15k/en_all_attrs_range
	python ent2vec_sparse.py ../../data/dbp15k/ja_en/ 0.3/del_atts_0_${i} 0.95 0.95 0.9
	python -u cse_pos_neg.py ../../data/dbp15k/ja_en/ 0.3/del_atts_0_${i} > output/ja_en_del_atts_0_${i}
done

for i in 1 2 3
do
	python attr2vec.py ../../data/dbp15k/fr_en/ ../../data/dbp15k/fr_en/0_3/nodes_0_${i}/ ../../data/dbp15k/fr_en/all_attrs_range ../../data/dbp15k/en_all_attrs_range
	python ent2vec_sparse.py ../../data/dbp15k/fr_en/ 0.3/nodes_0_${i} 0.95 0.95 0.9
	python -u cse_pos_neg.py ../../data/dbp15k/fr_en/ 0.3/nodes_0_${i} > output/fren_delnodes_${i}
done
# 
for i in 1 2 3
do
	python attr2vec.py ../../data/dbp15k/zh_en/ ../../data/dbp15k/zh_en/0_3/edges_0_${i}/ ../../data/dbp15k/zh_en/all_attrs_range ../../data/dbp15k/en_all_attrs_range
	python ent2vec_sparse.py ../../data/dbp15k/zh_en/ 0.3/edges_0_${i} 0.95 0.95 0.9
	python -u cse_pos_neg.py ../../data/dbp15k/zh_en/ 0.3/edges_0_${i} > output/zhen_deledges_${i}
done

for i in 1 2 3
do
	python attr2vec.py ../../data/dbp15k/ja_en/ ../../data/dbp15k/ja_en/0_3/edges_0_${i}/ ../../data/dbp15k/ja_en/all_attrs_range ../../data/dbp15k/en_all_attrs_range
	python ent2vec_sparse.py ../../data/dbp15k/ja_en/ 0.3/edges_0_${i} 0.95 0.95 0.9
	python -u cse_pos_neg.py ../../data/dbp15k/ja_en/ 0.3/edges_0_${i} > output/jaen_deledges_${i}
done

for i in 1 2 3
do
	python attr2vec.py ../../data/dbp15k/fr_en/ ../../data/dbp15k/fr_en/0_3/edges_0_${i}/ ../../data/dbp15k/fr_en/all_attrs_range ../../data/dbp15k/en_all_attrs_range
	python ent2vec_sparse.py ../../data/dbp15k/fr_en/ 0.3/edges_0_${i} 0.95 0.95 0.9
	python -u cse_pos_neg.py ../../data/dbp15k/fr_en/ 0.3/edges_0_${i} > output/fren_deledges_${i}
done
#




#
python train.py
cd ../Crosslingula-KG-Matching/
python run_model.py



for i in 1 2 3 4 5
do
	mkdir nodes_0_${i}
	cp create_graph.py nodes_0_${i}/.
	cd nodes_0_${i}
	cp ../triples_2 ./
	cp ../sup_rel_ids ./
	cp ../rel_ids_1 ./
	cp ../rel_ids_2 ./
	cp ../ent_ids_2 ./
	cd ../
done

for i in 1 2 3 4 5
do
	mkdir edges_0_${i}
	cp noise_edges.py edges_0_${i}/.
	cd edges_0_${i}
	cp ../ref_ents ./
	cp ../ref_ent_ids ./
	cp ../sup_ent_ids ./
	cp ../triples_2 ./
	cp ../sup_rel_ids ./
	cp ../rel_ids_1 ./
	cp ../rel_ids_2 ./
	cp ../ent_ids_2 ./
	cp ../ent_ids_1 ./
	cd ../
done
