
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import random 

class KMEANSClustering(BaseEstimator,ClassifierMixin):

    def __init__(self,k=3,debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.X = X
        self.centroids = self.X[:self.k,:] if self.debug==True else self.X[tuple(random.sample(range(X.shape[0]), self.k)),:]
        self.clusters = [ [] for i in range(self.k)]   

        self.newCentroids = self.centroids+1
        self.terninationCondition = [ np.linalg.norm(self.newCentroids[i]-self.centroids[i]) <0.1 for i in range(self.centroids.shape[0])]
        while all(self.terninationCondition):
            self.group_points()
            self.update_centroids()

        return self
    
    def group_points(self):
        self.clusters = [ [] for i in range(self.k)] 
        for i, x in enumerate(self.X):
            index_ = self.find_closest_centroid(x)
            self.clusters[index_].append[i]

    def update_centroids(self):
        for i in range(self.k):
            data = self.X[tuple(self.clusters[i]),:]
            self.newCentroids[i] = np.mean(data, axis=0)
        self.terninationCondition = [ np.linalg.norm(self.newCentroids[i]-self.centroids[i]) <0.1 for i in range(self.centroids.shape[0])]
        self.centroids = self.newCentroids

    def find_closest_centroid(self, x):
        minLen = np.inf
        minIndex = -1
        for i, centroid in enumerate(self.centroids):
            length_ = np.linalg.norm(x-centroid)
            if length_ < minLen:
                minLen = length_
                minIndex = i
        return minIndex        
    
    def print_clusters(self):
        """
            Used for grading.
            print("Num clusters: {:d}\n".format(k))
            print("Silhouette score: {:.4f}\n\n".format(silhouette_score))
            for each cluster and centroid:
                print(np.array2string(centroid,precision=4,separator=","))
                print("{:d}\n".format(size of cluster))
        """
        pass