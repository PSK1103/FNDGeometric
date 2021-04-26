import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import json
import time
from twython import Twython, TwythonError
from config import DATA_DIR, DEVICE
import tf_geometric as tfg
import tensorflow as tf
from util.objects import User
from twitter_keys import APP_SECRET, APP_KEY, ACCESS_TOKEN
if ACCESS_TOKEN == '':
    twitter = Twython(APP_KEY,APP_SECRET)
    ACCESS_TOKEN = twitter.obtain_access_token()
    f = open('twitter_keys.py','w')
    f.write("APP_KEY = '{}'\n".format(APP_KEY))
    f.write("APP_SECRET = '{}'\n".format(APP_SECRET))
    f.write("ACCESS_TOKEN = '{}'\n".format(ACCESS_TOKEN))
    f.close()
twitter = Twython(app_key=APP_KEY,access_token=ACCESS_TOKEN)

def getUser(tweet_id):
    """
    Get the tweeter of tweet corresponding to tweet id

    Args:
        tweet_id (str): id of tweet

    Returns:
        user (User): details of tweeter
    """
    tweet = twitter.show_status(id=tweet_id)
    user = tweet['user']
    return User(user['id'],user['verified'])

def getFollowers(uid):
    """get list of followers of a user"""
    return twitter.get_followers_ids(user_id=str(uid))['ids']

def getFollowing(uid):
    """get list of ids of users that the user follows"""
    return twitter.get_friends_ids(user_id=str(uid))['ids']

def createUserDB():
    """Creates a torch_geometric Data object

    Returns:
        dataset (Data): torch_geometric Data object
    """        
    if not os.path.exists(DATA_DIR + '/twt2user'):
        print('Creating new dataset')
        users = {}
        gossipcop_fake = pd.read_csv(DATA_DIR + '/gossipcop_fake.csv')
        gossipcop_fake.dropna()
        gossipcop_fake = gossipcop_fake
        gossipcop_real = pd.read_csv(DATA_DIR + '/gossipcop_real.csv')
        gossipcop_real.dropna()
        politifact_real = pd.read_csv(DATA_DIR+ '/politifact_real.csv')
        politifact_real.dropna()
        politifact_fake = pd.read_csv(DATA_DIR + '/politifact_fake.csv')
        politifact_fake.dropna()
        acc = 0
        twt2user = open(DATA_DIR + '/twt2user','a')

        count = 0
        for tweets in gossipcop_fake['tweet_ids']:
            if isinstance(tweets,float):
                tweets = [tweets]
            for id in tweets.split():
                if count >= 100000:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    break
                acc += 1
                if acc > 300:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    print('waiting for api cooldown')
                    time.sleep(15*60)
                    acc = 1
                try:
                    user = getUser(tweet_id=id)
                    twt2user.write('{} {} {} {}\n'.format(id,user.uid,int(user.verified),1))
                    if user.uid not in users:
                        print('adding user with id: ',user.uid)
                        users[user.uid] = user
                    users[user.uid].real_news()
                except:
                    pass

        count = 0
        for tweets in gossipcop_real['tweet_ids']:
            if isinstance(tweets,float):
                tweets = [tweets]
            for id in tweets.split():
                if count >= 100000:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    break
                acc += 1
                if acc > 300:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    print('waiting for api cooldown')
                    time.sleep(15*60)
                    acc = 1
                try:
                    user = getUser(tweet_id=id)
                    twt2user.write('{} {} {} {}\n'.format(id,user.uid,int(user.verified),1))
                    if user.uid not in users:
                        print('adding user with id: ',user.uid)
                        users[user.uid] = user
                    users[user.uid].real_news()
                except:
                    pass
        count = 0
        for tweets in politifact_fake['tweet_ids']:
            if isinstance(tweets,float):
                tweets = [tweets]
            for id in tweets.split():
                if count >= 100000:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    break
                acc += 1
                if acc > 300:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    print('waiting for api cooldown')
                    time.sleep(15*60)
                    acc = 1
                try:
                    user = getUser(tweet_id=id)
                    twt2user.write('{} {} {} {}\n'.format(id,user.uid,int(user.verified),1))
                    if user.uid not in users:
                        print('adding user with id: ',user.uid)
                        users[user.uid] = user
                    users[user.uid].real_news()
                except:
                    pass

        count = 0    
        for tweets in politifact_real['tweet_ids']:
            if isinstance(tweets,float) & tweets != 0.0:
                tweets = [tweets]
            for id in tweets.split():
                if count >= 100000:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    break
                acc += 1
                if acc > 300:
                    twt2user.close()
                    twt2user = open(DATA_DIR + '/twt2user','a')
                    print('waiting for api cooldown')
                    time.sleep(15*60)
                    acc = 1
                try:
                    user = getUser(tweet_id=id)
                    twt2user.write('{} {} {} {}\n'.format(id,user.uid,int(user.verified),1))
                    if user.uid not in users:
                        print('adding user with id: ',user.uid)
                        users[user.uid] = user
                    users[user.uid].real_news()
                except:
                    pass
        twt2user.close()

        users_list = [user for _,user in users.items()]
        print('Total {} users in dataset'.format(len(users_list)))
        user_data = open(DATA_DIR + '/users','a')
        i = 0            
        first = True
        for user in users_list:
            if first:
                first = False
            else:
                user_data.write('\n')
            i += 1
            if i>=15:
                user_data.close()
                user_data = open(DATA_DIR + '/users','a')
                print('waiting for api cooldown')
                time.sleep(15*60)
            user.compute_rating()
            user.set_followers(getFollowers(user.uid))
            user.set_following(getFollowing(user.uid))
            user_data.write('{},{},{},{}'.format(user.uid,len(user.followers),len(user.following),user.verified))
        user_data.close()
    
    elif not os.path.exists(DATA_DIR + '/users'):
        twt2user = open(DATA_DIR + '/twt2user','r')
        print('found twt2user')
        print('verifying all tweet data downloaded')
        users = {}
        for line in twt2user.readlines():
            try:
                data = line.split()
                user = User(uid=data[1],verified=data[2])
                if user.uid not in users:
                    users[user.uid] = user
                if data[3] == 1:
                    users[user.uid].real_news()
                else:
                    users[user.uid].fake_news()
            except:
                pass

        users_list = [user for _,user in users.items()]
        print('Total {} users in dataset'.format(len(users_list)))
        user_data = open(DATA_DIR + '/users','a')
        i = 0            
        first = True
        for user in users_list:
            if first:
                first = False
            else:
                user_data.write('\n')
            i += 1
            if i>=15:
                user_data.close()
                user_data = open(DATA_DIR + '/users','a')
                print('waiting for api cooldown')
                time.sleep(15*60)
            user.compute_rating()
            user.set_followers(getFollowers(user.uid))
            user.set_following(getFollowing(user.uid))
            user_data.write('{},{},{},{},{}'.format(str(user.uid),str(user.verified),str(user.rating),' '.join([str(id) for id in user.followers]),' '.join(str(id) for id in user.following)))
        user_data.close()

    createDatasetFiles()

    # x,y,edge_index = createGraphTensors(users_list)

    # dataset = tfg.Graph(x=x,y=y, edge_index=edge_index)
    # dataset.convert_data_to_tensor()
    # outputs = tfg.layers.GCN(units=dataset.num_nodes,activation=tf.nn.relu,)
    return None
    
def createDatasetFiles():
    """
    creates tensors to create a Dataset object

    Args:
        users (list): list of User objects

    Returns:
        x (tensor): tensor of features of each node
        y (tensor): tensor of labels of each node
        edges (tensor): tensor containing edge indices in the COO format
    """

    users_list = []

    users_data = open(DATA_DIR + '/users','r')
    for line in users_data.readlines():
        data = line.split(',')
        user = User(int(data[0]),int(data[1]),rating=float(data[2]),followers=[int(id) for id in data[3].split()],following=[int(id) for id in data[4].split()])
        users_list.append(user)

    idx2uid = {}
    uid2idx = {}
    for i,uid in enumerate(users_list):
        idx2uid[i+1] = uid
        uid2idx[uid] = i+1

    FND_A = open(DATA_DIR + '/FND_A.txt','w')
    FND_gi = open(DATA_DIR + '/FND_graph_indicator.txt','w')
    FND_na = open(DATA_DIR + '/FND_node_attributes.txt','w')
    FND_nl = open(DATA_DIR + '/FND_node_labels.txt','w')
    first = True

    for user in users_list:
        if first:
            first = False
        else:
            FND_gi.write('\n')
            FND_A.write('\n')
            FND_na.write('\n')
            FND_nl.write('\n')

        for id in user.followers + user.following:
            if id in uid2idx:   
                FND_A.write('{}, {}\n'.format(uid2idx[user.uid],uid2idx[uid]))
                FND_A.write('{}, {}'.format(uid2idx[uid],uid2idx[user.uid]))
        FND_na.write('{}, {}, {}, {}'.format(user.rating,len(user.following),len(user.followers),int(user.verified)))
        FND_gi.write('1')
        FND_nl.write(str(int(user.rating>0)))

        
    FND_gi.close()
    FND_A.close()

    FND_gl = open(DATA_DIR + '/FND_graph_labels.txt','w')
    FND_gl.write('1')
    FND_gl.close()


if __name__ == '__main__':
    createUserDB()