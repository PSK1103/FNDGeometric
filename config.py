DEVICE = 'cuda'

DATA_DIR = 'dataset'
BATCH_SIZE = 64

LR = 0.001
WEIGHT_DECAY = 0.01
HIDDEN = 128
EPOCHS = 100

BUZZFEED_DATA = [
    'BuzzFeedNews.txt',
    'BuzzFeedUser.txt',
    'BuzzFeedNewsUser.txt',
    'BuzzFeedUserUser.txt',
    'BuzzFeedUserFeature.mat',
]
POLITIFACT_DATA = [
    'PolitiFactNews.txt',
    'PolitiFactUser.txt',
    'PolitiFactNewsUser.txt',
    'PolitiFactUserUser.txt',
    'PolitiFactUserFeature.mat',
]