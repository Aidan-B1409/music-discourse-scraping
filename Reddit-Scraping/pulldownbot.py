import praw
import csv
from datetime import datetime
from pandas import DataFrame, read_csv
import pandas as pd
from query import Query

reddit = praw.Reddit('bot1')
"""TODO - Make megasubreddit object based on subreddits.txt"""

def readQueries():
    queries = list()
    file = r'test.csv'
    df = pd.read_csv(file)[['artist_name', 'track_name']]
    ##print(df.head(5))
    for index, row in df.iterrows():
        print(row['artist_name'], row['track_name'])
        query = Query(row['artist_name'], row['track_name'], reddit);
        mineComments(query.getSubmissions(), query);

def mineComments(resultsList, query):
    """counter for CSV rows"""
    n = 0
    """Time of CSV initialization"""
    dtime_string = datetime.now().strftime('%d-%m-%Y%H-%M-%S')
    with open('reddit_'+dtime_string+'.csv', 'w', encoding='utf-8', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fileWriter.writerow(["Comment Index", "Submission ID", "Submission Title", "Comment Body", "Comment Score", "Query", "Subreddit Name", "Comment ID"])

        for submission in resultsList:
            for comment in submission.comments:
                submissionID = submission.id
                submissionTitle = submission.title
                commentBody = comment.body.replace(",","").encode('utf-8', "ignore");
                commentBody = str(commentBody, encoding = 'utf-8');
                commentScore = comment.score
                entryQuery = query.query
                subredditName = comment.subreddit.display_name
                print(subredditName)
                commentID = comment.id

                fileWriter.writerow([n, submissionID, submissionTitle, commentBody, commentScore, entryQuery, subredditName, commentID])
                n = n + 1



if __name__ == '__main__':
    readQueries()
    #query = Query("Madeon", "All My Friends", reddit);
    #mineComments(query.getSubmissions())