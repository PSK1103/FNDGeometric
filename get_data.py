import pandas as pd
import numpy as np
import os
import time
from config import DATA_DIR, DEVICE
from config import POLITIFACT_DATA as files
from stellargraph import IndexedArray, StellarDiGraph
import scipy.io as sio

class User:
    def __init__(self,uid):
        self.uid = uid
        self.score = 0
    def update(self,i):
        self.score += i

def createGraph():
    news = open(os.path.join(DATA_DIR,files[0]),'r')
    news = news.readlines()

    users = open(os.path.join(DATA_DIR,files[1]),'r')
    user_objs = [User(uid.strip()) for uid in users.readlines()]
    users.close()

    news_user = open(os.path.join(DATA_DIR,files[2]),'r')
    for line in news_user.readlines():
        data = [int(i) for i in line.split()]
        user_objs[data[1] - 1].update(data[2] if 'Real' in news[data[0]-1] else -data[2])
    news_user.close()

    # user_objs = user_objs[:1000]

    node_labels = {}
    for user in user_objs:
        node_labels[user.uid] = int(user.score>0)
    nodes = [user.uid for user in user_objs]

    user_edges = open(os.path.join(DATA_DIR,files[3]),'r')
    edges = [[int(num) for num in line.split()] for line in user_edges.readlines()]
    user_edges.close()
    # eedges = []
    # for edge in edges:
    #     if(edge[0]<1000 and edge[1]<1000):
    #         eedges.append(edge)
    # edges = eedges

    edges = [[user_objs[i-1].uid for i in edge] for edge in edges]
    edges = np.transpose(edges)

    node_features = sio.loadmat(os.path.join(DATA_DIR,files[4]))['X'].toarray()#[:1000]
    node_features = np.transpose(node_features)
    node_features = node_features[np.random.choice(node_features.shape[0], 2000, replace=False)]
    node_features = np.transpose(node_features)

    nodes = IndexedArray(node_features,nodes)
    
    edges = pd.DataFrame({
        'source':edges[0],'target':edges[1]
    })
    sg = StellarDiGraph(nodes,edges)
    node_labels = pd.Series(data=node_labels,index=None)
    print(sg.info())
    return sg,node_labels

if __name__ == '__main__':
    createGraph()