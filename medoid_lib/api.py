# data type is DataFrame, with a cloumn called label
def categorize_data(data, label_col='Label', limit=36 * 5, path='./categorized_data/', store='true'):
   labels = set(data[label_col].values)
   new_labels = []
   categorized_data = []
   for label in labels:
      tmp = data.loc[data[label_col] == label]
      if tmp.shape[0] < limit:
         print('the data with label:' + label + ' does not have enough data, dropped')
         continue
      new_labels.append(str(label))
      categorized_data.append((tmp.drop(['Label'], axis=1), label))
      if store:
         tmp.to_csv(path + str(label) + '.csv', index=False)
   return new_labels, categorized_data

def create_segs(data, duration):
   result = data.values.tolist()
   segs = []
   length = len(result)
   for i in range(0, length, duration):
      if i + duration <= length:
         segs.append(result[i:i + duration])
   return segs

import numpy as np
# from dtw_lib import _dtw_lib
# from scipy.spatial.distance import euclidean

# def distance(seg1, seg2, relax):
   # distance, path, D = _dtw_lib.fastdtw(seg1, seg2, relax=relax, dist=euclidean)
   # return distance

import pickle
def store(files, names):
   for i in range(len(files)):
      pickle_out = open('data/' + names[i], "wb")
      pickle.dump(files[i], pickle_out)
      pickle_out.close()
        
def load(names):
    result = []
    for i in names:
        pickle_in = open(i, 'rb')
        result.append(pickle.load(pickle_in))
    return result

def calculate_matrix(segs, distance):
   length = len(segs)
   table = [[-1 for i in range(length)] for i in range(length)] # initialize the table to all -1
   for i in range(length):
      for j in range(length):
         if i == j: 
            table[i][j] = 0
            continue
         elif table[j][i] != -1:  # using memoization
            table[i][j] = table[j][i]
         else:
            table[i][j] = distance(segs[i], segs[j])
   return table

def plot_matrix(matrix, title, tick_labels=None, tag=False, name="", save=False):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    # set the ticks
    if tag:
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_yticks(np.arange(len(tick_labels)))
        # set entry with labels
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    title = "distance distribution for %s " % title 
    ax.set_title(title)
    fig.tight_layout()
    # scale the range of the bar to be 0 - 60
    quadmesh = ax.pcolormesh(matrix)
    quadmesh.set_clim(0,60)
    fig.colorbar(quadmesh)
    if save: plt.savefig(name)
    plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage  
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
def hieratical_plot(matrix, segs, link, title, name="", save=False):
    f = lambda x, y: matrix[x[1]][y[1]]
    X = list(map(lambda x: [0, x], range(len(segs))))
    Y = pdist(X, f)
    linked = linkage(Y, link, metric = '')

    labelList = range(0, len(segs))

    plt.figure(figsize=(10, 7))  
    dendrogram(linked,  
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.title(title + "======" + link)
    if save: plt.savefig(name)
    plt.show()  
    
from sklearn.cluster import AgglomerativeClustering
def get_hieratical_cluster(matrix, num_cluster, linkage='complete'):
   cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='precomputed', linkage=linkage)  
   return cluster.fit_predict(matrix)

def convert_label_to_clusters(l):
   clusters = [[] for i in range(max(l) + 1)]
   for indx, value in enumerate(l):
      clusters[value].append(indx)
   return clusters

def get_medoid(cluster, matrix, segs):
   """get the medoids from one cluster

   calculate medoids with cluster(index) distance matrix, and the segments

    Args:
      cluster: the index of the segs in the cluster
      matrix: the distance matrix base on fast_relax dtw, of the same label.
      segs: the segments of the current label.
    Returns:
      the tuples which contains the (medoids, D) where D is the average
      distance from medoid to other segs in the cluster. 
    """
   dis = [0 for _ in range(len(cluster))]
   for i in range(len(cluster)):
      for j in cluster:
         dis[i] += matrix[cluster[i]][j]
   index = np.argmax(dis)
   return segs[index], dis[index] / len(cluster)

def get_medoids(matrix, segs, num, label):
   """get the medoids with distance matrix, and num of medoids

    Args:
      matrix: the distance matrix base on fast_relax dtw, of the same label.
      segs: the segments of the current label.
      num: the number of the medoids.
    Returns:
      the list of tuples which contains the (medoids, D) where D is the average
      distance from medoid to other segs in the cluster. 
    """
   clusters = convert_label_to_clusters(get_hieratical_cluster(matrix, num))
   medoids = []
   for c in clusters:
      medoids.append(get_medoid(c, matrix, segs) + (label,))
   return medoids

def classify(represents, seg, distance, top=1):
   """classify the unknown seg base on the medoids

    each activity get multiple medoids, which are seg, and classify the new seg base on the 
    distance from the seg and the medoids

    Args:
        represents: a list of of represent, each represent is a 3-dimension tuple, which contains
            (medoids, D, Label), where D is the average value from the medoid to other segs in the 
            cluster
        seg: unknown seg, in the same duration as the data
        top: to get the top n predictions, default is 1

    Returns:
        A list which contains the predication of the unknown seg.
    """
   dis_f = lambda rep: distance(rep[0], seg) / rep[1]
   result = sorted(represents, key=dis_f)
   result = list(map(lambda rep: rep[-1], result))
   return result[:top]

def evaluate(test_segs, test_label, represents, distance, top):
   predict = list(map(lambda seg: classify(represents, seg, distance, top), test_segs))
   result = []
   for i in range(len(predict)):
      result.append(test_label[i] in predict[i])
   return np.mean(result)