import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# load population
with open("../experiments/high_g_enemy=2_27092024_170302_0/final_population.pkl", mode="rb") as pkl_file:
    final_pop_small = pickle.load(pkl_file)

# load population
with open("../experiments/low_g_enemy=2_27092024_183050_0/final_population.pkl", mode="rb") as pkl_file:
    final_pop_big = pickle.load(pkl_file)

from sklearn.metrics import silhouette_score

best_score = -1
best_k = None


for k in range(2, 50):  # Silhouette doesn't support 1 cluster
    silhouette_scores = []
    for _ in range(5):
        kmeans = KMeans(n_clusters=k).fit(final_pop_big)
        labels = kmeans.labels_
        score = silhouette_score(final_pop_big, labels)
        silhouette_scores.append(score)
    avg_score = np.array(silhouette_scores).mean()

    if avg_score > best_score:
        best_score = avg_score
        best_k = k

print(f"(BIG): The optimal number of clusters is: {best_k} with silhouette score: {best_score}")

best_score = -1
best_k = None

for k in range(2, 50):  # Silhouette doesn't support 1 cluster
    silhouette_scores = []

    for _ in range(5):
        kmeans = KMeans(n_clusters=k).fit(final_pop_small)
        labels = kmeans.labels_
        score = silhouette_score(final_pop_small, labels)
        silhouette_scores.append(score)
    avg_score = np.array(silhouette_scores).mean()
    
    if avg_score > best_score:
        best_score = avg_score
        best_k = k

print(f"(SMALL): The optimal number of clusters is: {best_k} with silhouette score: {best_score}")

