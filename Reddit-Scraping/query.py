import praw
import csv
from datetime import datetime

class Query:
    query = "null"
    SEARCHLIMIT = 3;

    def __init__(self, track, artist, reddit):
        self.track = track
        self.artist = artist
        self.reddit = reddit
        self.subreddit = reddit.subreddit("all")
        self.query = Query.buildQuery(self, track, artist)

    def buildQuery(self, track, artist):
        return str("title:" + "\"" + track + "\"" + " " +  "\"" + artist + "\"")

    def getSubmissions(self):
        submissionList = list()
        subreddit = self.reddit.subreddit("all")
        for submission in subreddit.search(self.query, 'top', 'lucene', "all", limit=Query.SEARCHLIMIT):
            submission.comments.replace_more()
            submissionList.append(submission)
        return submissionList
