import pandas as pd
from util.objects import User
from twython import Twython, TwythonError
import time
DATA_DIR = 'dataset'

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

users = {}
gossipcop_real = pd.read_csv('dataset/gossipcop_real.csv')
gossipcop_real.dropna()
acc = 0

count = 0
for tweets in gossipcop_real['tweet_ids']:
    if isinstance(tweets,float):
        continue
    for id in tweets.split():
        if count >= 100000:
            twt2user.close()
            twt2user = open(DATA_DIR + '/twt2user','a')
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