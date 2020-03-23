import praw

import pandas as pd
from query import Query

reddit = praw.Reddit('bot1')
def readQueries():
    file = r'test.csv'
    df = pd.read_csv(file)
    for index, row in df.iterrows():
        print(row['artist_name'], row['track_name'])
        query = Query(row['artist_name'], row['track_name'], reddit)
        valence = row['valence']
        arousal = row['arousal']
        deezerID = row['dzr_sng_id']
        query.mineComments(index, valence, arousal, deezerID)

if __name__ == '__main__':
    readQueries()