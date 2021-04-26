class User:
    def __init__(self,uid,verified,rating=0.0,followers=[],following=[]):
        self.uid = uid
        self.verified = verified
        self.rating = rating
        self.tweets = 0.0
        self.score = 0.0
        self.followers = followers
        self.following = following
        
    def set_followers(self,followers):
        self.followers = followers

    def set_following(self,following):
        self.following = following
    
    def real_news(self):
        self.score += 1
        self.tweets += 1

    def fake_news(self):
        self.score -=1
        self.tweets += 1
    
    def compute_rating(self):
        self.rating = self.score/self.tweets

    