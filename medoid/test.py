activities = ['Walking', 'Running', 'Commute in bus', 'Eating using fork and spoon', 
             'Using mobile phone(texting)', 'Working on laptop', 'Sitting', 'Washing hands',
             'Eating with hand', 'Conversing while sitting', 'Elevator', 'Opening door',
             'Standing', 'Climbing upstairs', 'Running']
columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Label']
import pandas as pd
import numpy as np
def get_raw_data(activity):
    return pd.read_csv("./categorized_data/" + activity + ".csv")
import random
from tqdm import tqdm
def totoal_a(accs):
    import numpy as np
#     for i in range(len(accs)):
#         accs[i] = (accs[i]  * 3.0 / 63.0 - 1.5) * 9.8
    return np.sqrt(np.sum(map(lambda x: np.square(x), accs)))
    
def create_segs(data, duration):
    data = data.iloc[:, :3] # get x, y, z acceleration
    result = []
    for index, row in data.iterrows():
        result.append(totoal_a(list(row)[:3]))
    segs = []
    length = len(result)
    for i in tqdm(range(0, length, 32 * duration)):
        if i + duration * 32 <= length:
            segs.append(result[i:i + duration * 32])
    return segs

import relaxed_dtw_v1
import numpy as np

def distance(seg1, seg2):
    result = 0.0;
    dist = lambda x,y : np.abs(x - y)
    distance, matrix = relaxed_dtw_v1.relaxed_dtw(seg1, seg2, distance=dist, r=16)
    return distance
        
def find_medoid_seg(segs):
    length = len(segs)
    result = [0 for i in range(length)]
    table = [[-1 for i in range(length)] for i in range(length)]
    for i in tqdm(range(length)):
        for j in range(length):
            if i == j: continue
            elif table[i][j] != -1:
                result[i] += table[i][j]
            else:
                table[i][j] = distance(segs[i], segs[j])
                result[i] = table[i][j]
    min_medoid = min(result)
    for i in range(len(result)):
        if min_medoid == result[i]:
            return segs[i], min_medoid

all = []
for i in random.sample(list(range(len(activities))), 4):
	segs = create_segs(get_raw_data(activities[i]), 5)
	all.extend(random.sample(segs, int(len(segs) / 4)))
table = [[-1 for i in range(len(all))] for j in range(len(all))]
for i in range(len(all)):
	for j in range(len(all)):
		if i == j: table[i][j] = 0
		elif i > j:
			table[i][j] = table[j][i]
		else:
			table[i][j] = distance(all[i], all[j])

import networkx as nx
import numpy as np
import string
import pickle
pickle_out = open("table.pickle","wb")
pickle.dump(table, pickle_out)
pickle_out.close()


dt = [('len', float)]
A = np.array(table) * 20
A = A.view(dt)

G = nx.from_numpy_matrix(A)
G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))    

G = nx.drawing.nx_agraph.to_agraph(G)

# G.node_attr.update(color="red", style="filled")
# G.edge_attr.update(color="blue", width="2.0")

G.draw('./out.png', format='png', prog='neato')
