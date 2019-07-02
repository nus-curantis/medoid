def create_segs(data, duration):
   data = data.iloc[:, :3] # get x, y, z acceleration
   result = data.values.tolist()
   segs = []
   length = len(result)
   for i in tqdm(range(0, length, hz * duration)):
      if i + duration * hz <= length:
         segs.append(result[i:i + duration * hz])
   return segs

import numpy as np
from dtw_lib import _dtw_lib
from scipy.spatial.distance import euclidean

def distance(seg1, seg2, relax):
   distance, path, D = _dtw_lib.fastdtw(seg1, seg2, relax=relax, dist=euclidean)
   return distance

import pickle
def store(files, names):
   for i in range(len(files)):
      pickle_out = open('data/' + names[i], "wb")
      pickle.dump(files[i], pickle_out)
      pickle_out.close()

def calculate_matrix(segs):
   length = len(segs)
   table = [[-1 for i in range(length)] for i in range(length)] # initialize the table to all -1
   for i in tqdm(range(length)):
      for j in range(length):
         if i == j: 
            table[i][j] = 0
            continue
         elif table[j][i] != -1:  # using memoization
            table[i][j] = table[j][i]
         else:
            table[i][j] = distance(segs[i], segs[j], 1)
   return table

from sklearn.cluster import AgglomerativeClustering
def get_hieratical_cluster(matrix, num_cluster):
   cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='precomputed', linkage='complete')  
   return cluster.fit_predict(matrix)

def convert_label_to_clusters(l):
   clusters = [[] for i in range(max(l) + 1)]
   for indx, value in enumerate(l):
      clusters[value].append(indx)
   return cluster
