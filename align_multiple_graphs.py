
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from collections import Counter


emb_path = "embeddings.tsv"
embeddings = np.loadtxt(emb_path, delimiter='\t')
print(embeddings)

label_path = "label.tsv"
label = []

with open(label_path, 'r', encoding='utf-8') as file:
    for line in file:
        data_line = line.strip()
        label.append(int(data_line))
label_count = Counter(label)
print(label_count)
max_len = max(list(label_count.values()))

# kmeans = KMeans()
kmeans = KMeans(n_clusters=max_len, random_state=0).fit(embeddings)
kmeans_label = kmeans.labels_
with open("k_means_labels.tsv", "w", encoding='utf-8') as file:
    file.write("graph_label"+ "\t" +"clus_label\n")
    for i in range(len(label)):
        file.write("{}\t{}\n".format(label[i], kmeans_label[i]))

kmeans_counter = Counter(kmeans_label)
print(kmeans_counter)
count_lol = 0
for ele, count in kmeans_counter.items():
    if count < 50:
        count_lol += 1
print(count_lol) 

# labels = kmeans.labels_
# with open("k_means_labels.tsv", "w", encoding='utf-8') as file:
#     for i in range(len(labels)):
#         file.write("{}\n".format(labels[i]))