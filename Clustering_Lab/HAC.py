
from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
from sklearn import cluster
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

class HACClustering(BaseEstimator,ClassifierMixin):

    def __init__(self,k=3,link_type='single', normalize=True): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.normalize = normalize
        
    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.X = X         
        self.X = np.append(self.X, y.reshape(len(y),-1), axis=1) if y else self.X
        if self.normalize==True: 
            self.normalize_data()
        self.clusters_ = [i for i in range(X.shape[0])]
        while len(self.clusters_)>self.k:
            i,j = self.find_closest_clusters()
            self.group_clusters(i,j)
        self.centroids = self.find_centroids()
        return self
    
    def print_clusters(self):    
        # Used for grading.
        print("Num clusters: {:d}\n".format(self.k))
        print("Silhouette score: {:.4f}\n\n".format(self.calculate_silhouette_score()))
        for i,centroid in  enumerate(self.centroids):
            print(np.array2string(centroid,precision=4,separator=","))
            print("{:d}\n".format(len(self.clusters_[i]) if type(self.clusters_[i])==list else 1))


    def group_clusters(self, idx1, idx2):
        self.clusters_[idx1] = self.clusters_[idx1] if type(self.clusters_[idx1])==list else [self.clusters_[idx1]] 
        self.clusters_[idx2] = self.clusters_[idx2] if type(self.clusters_[idx2])==list else [self.clusters_[idx2]] 
        self.clusters_[idx1] = self.clusters_[idx1] + self.clusters_[idx2]
        del self.clusters_[idx2]

    def find_closest_clusters(self):
        size = len(self.clusters_)
        Matrix = np.ones((size,size))*np.inf
        for i in range(size):
            for j in range(size):
                if j>i:
                    Matrix[i,j]=self.find_length_between_two_clusters(i,j)
        
        minValue = np.amin(Matrix)
        res = np.where(Matrix==minValue)
        row = res[0][0]
        col = res[1][0]
        return row, col

    def find_length_between_two_clusters(self, idx1, idx2):
        indexes1 = tuple(self.clusters_[idx1]) if type(self.clusters_[idx1])==list else self.clusters_[idx1]
        Mat1 = self.X[indexes1,:]
        indexes2 = tuple(self.clusters_[idx2]) if type(self.clusters_[idx2])==list else self.clusters_[idx2]
        Mat2 = self.X[indexes2,:]

        if self.link_type=='single':
            length_ = self.find_length_between_two_clusters_for_single_link_type(Mat1,Mat2)
        elif self.link_type == 'complete':
            length_ = self.find_length_between_two_clusters_for_complete_link_type(Mat1, Mat2)  
        return length_

    def find_length_between_two_clusters_for_single_link_type(self,Mat1, Mat2):
        lenght_ = np.inf
        if Mat1.ndim == 1 and Mat2.ndim == 1:
            if np.linalg.norm(Mat1-Mat2) < lenght_ :
                    lenght_= np.linalg.norm(Mat1-Mat2)                    
        elif Mat1.ndim == 1 and Mat2.ndim > 1:
            for vec in Mat2:
                if np.linalg.norm(Mat1-vec) < lenght_ :
                    lenght_= np.linalg.norm(Mat1-vec)
        elif Mat1.ndim > 1 and Mat2.ndim == 1:
            for vec in Mat1:
                if np.linalg.norm(vec-Mat2) < lenght_ :
                    lenght_= np.linalg.norm(vec-Mat2)
        else:
            for vec1 in Mat1:
                for vec2 in Mat2:
                    if np.linalg.norm(vec1-vec2) < lenght_:
                        lenght_= np.linalg.norm(vec1-vec2)
        return lenght_

    def find_length_between_two_clusters_for_complete_link_type(self, Mat1, Mat2):
        lenght_=0

        if Mat1.ndim == 1 and Mat2.ndim == 1:
            if np.linalg.norm(Mat1-Mat2) > lenght_:
                    lenght_= np.linalg.norm(Mat1-Mat2)
        elif Mat1.ndim == 1 and Mat2.ndim > 1:
            for vec in Mat2:
                if np.linalg.norm(Mat1-vec) > lenght_ :
                    lenght_= np.linalg.norm(Mat1-vec)
        elif Mat1.ndim > 1 and Mat2.ndim == 1:
            for vec in Mat1:
                if np.linalg.norm(vec-Mat2) > lenght_:
                    lenght_= np.linalg.norm(vec-Mat2)
        else:
            for vec1 in Mat1:
                for vec2 in Mat2: 
                    if np.linalg.norm(vec1-vec2) > lenght_:
                        lenght_= np.linalg.norm(vec1-vec2)    
        return lenght_

    def normalize_data(self):
        for i in range(self.X.shape[1]):
            self.X[:,i] = self.X[:,i] - np.amin(self.X[:,i])
            self.X[:,i] = self.X[:,i]/(np.amax(self.X[:,i])-np.amin(self.X[:,i]))

    def find_centroids(self):
        centroids = []
        for cluster in self.clusters_:
            data = self.X[tuple(cluster) if type(cluster)==list else cluster,:]            
            centroid = np.mean(data, axis=0) if data.ndim>1 else data
            centroids.append(centroid)
        return centroids

    def calculate_silhouette_score(self):
        clusterer = AgglomerativeClustering(n_clusters=self.k, linkage=self.link_type).fit(self.X)
        cluster_labels = clusterer.labels_
        score = silhouette_score(self.X, cluster_labels)
        return score
