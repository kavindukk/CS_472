from re import L
from scipy.io import arff
import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering, KMeans
from HAC import HACClustering

data = arff.loadarff('abalone.arff')
df = pd.DataFrame(data[0])
X = df.to_numpy()

# HACObject = HACClustering(k=5)
# self_ = HACObject.fit(X).print_clusters()

HACObject = HACClustering(k=5, link_type='complete')
self_ = HACObject.fit(X).print_clusters()


# def normalize_data(X):
#     for i in range(X.shape[1]):
#         X[:,i] = X[:,i] - np.amin(X[:,i])
#         X[:,i] = X[:,i]/(np.amax(X[:,i])-np.amin(X[:,i]))
#     return X
# X_normalized = normalize_data(X)
# clusterer = AgglomerativeClustering(n_clusters=5, linkage='single').fit(X_normalized)
# cluster_labels = clusterer.labels_
# score = silhouette_score(X, cluster_labels)
# print(score)