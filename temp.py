import pandas as pd
import numpy as np
import os
import time
from twython import Twython, TwythonError
from config import DATA_DIR, DEVICE
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

users = {}
gossipcop_real = pd.read_csv(DATA_DIR + '/gossipcop_real.csv')
gossipcop_real.dropna()
acc = 0
twt2user = open(DATA_DIR + '/twt2user2','a')
if os.path.exists(DATA_DIR + "/twt2user2"):
    twt2user_val = open(DATA_DIR + '/twt2user2','r')
    print('verifying tweet data')
val_twt_ids = [line.split()[0] for line in twt2user_val.readlines()]
iter = 0
count = 0
for tweets in gossipcop_real['tweet_ids']:
    if isinstance(tweets,float):
        tweets = str(tweets)
    for id in tweets.split():
        if count >= 20000:
            twt2user.close()
            twt2user = open(DATA_DIR + '/twt2user2','a')
            break
        
        if iter < len(val_twt_ids):
            if val_twt_ids[iter] == id:
                print('Found tweet data from tweet with id {}'.format(id))
                iter += 1
                continue
        acc += 1
        if acc > 300:
            twt2user.close()
            twt2user = open(DATA_DIR + '/twt2user2','a')
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
            count += 1
        except:
            pass
