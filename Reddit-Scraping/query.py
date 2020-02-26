import praw
import csv
from datetime import datetime

class Query:
    query = "null"
    submissionList = list()
    SEARCHLIMIT = 3;

    def __init__(self, track, artist, reddit):
        self.track = track
        self.artist = artist
        self.reddit = reddit
        self.subreddit = reddit.subreddit("all");
        self.query = Query.buildQuery(track, artist)

    def buildQuery(self, track, artist):
        return track + artist

    def getSubmissions(self):
        for submission in Query.subreddit.search(Query.query, 'top', 'lucene', "all", limit=Query.SEARCHLIMIT):
            submission.comments.replace_more()
            self.submissionList.append(submission)
