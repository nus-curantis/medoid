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
columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Label']
hz = 36
all_activities = load(['data/all_activities'])[0]

import pandas as pd
def get_raw_data(user, activity):
    return pd.read_csv('./categorized_data/user' + str(user) + '_' + activity + '.csv')

from tqdm import tqdm
    
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
        
def find_medoid_seg(segs, user, activity, duration):
    length = len(segs)
    result = [0 for i in range(length)]
    table = [[-1 for i in range(length)] for i in range(length)] # initialize the table to all -1
    for i in tqdm(range(length)):
        for j in range(length):
            if i == j: 
                table[i][j] = 0
                continue
            elif table[j][i] != -1:  # using memoization
                table[i][j] = table[j][i]
                result[i] += table[i][j]
            else:
                table[i][j] = distance(segs[i], segs[j], 1)
                result[i] += table[i][j]
    store([table], ['user' + str(user) + '_' + activity + '_matrix_' + str(duration)])
    min_medoid = min(result)
    for i in range(len(result)):
        if min_medoid == result[i]:
            return segs[i], min_medoid / len(segs)

def calculate_medoids(user, activities, duration):
    import random
    all_segments = []
    medoids = []
    represents = []
    user_str = 'user' + str(user) + '_'
    for i in range(len(activities)):
        segs = create_segs(get_raw_data(user, activities[i]), duration)
        all_segments.append(segs)
        seg, medoid = find_medoid_seg(segs, user, activities[i], duration)
        represents.append(seg)
        medoids.append(medoid)
        name = [user_str + activities[i] + "_segments_" + str(duration)]
        store([segs], name)
    names = [user_str + "all_segments_" + str(duration), user_str + "medoids_" + str(duration), user_str + "represents_" + str(duration)]
    store([all_segments, medoids, represents], names)
    return all_segments, medoids, represents

user = 2
calculate_medoids(user, all_activities[user][6:], 5)