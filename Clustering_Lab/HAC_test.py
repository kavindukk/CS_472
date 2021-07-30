from scipy.io import arff
import pandas as pd
import numpy as np

from HAC import HACClustering

data = arff.loadarff('abalone.arff')
df = pd.DataFrame(data[0])
X = df.to_numpy()

HACObject = HACClustering(k=5)
self_ = HACObject.fit(X)
