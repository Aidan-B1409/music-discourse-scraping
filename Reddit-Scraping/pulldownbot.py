import praw
import csv
from datetime import datetime

reddit = praw.Reddit('bot1')
"""TODO - Make megasubreddit object based on subreddits.txt"""
subreddit = reddit.subreddit("music")
"""TODO - Automate searching of multiple queries based on txt object"""
query = 'Madeon'
"""Very important constant, defines how deep the getSubmissions method will search"""
SEARCHLIMIT = 3

def getSubmissions(keyword):
    searchResults = list()
    for submission in subreddit.search(keyword, 'relevance', 'lucene', 'all',limit=SEARCHLIMIT):
        searchResults.append(submission)
    return searchResults
def mineComments(resultsList):
    """counter for entries into CSV"""
    n = 0
    """Time of CSV initialization"""
    dtime_string = datetime.now().strftime('%d/%m/%Y%H:%M:%S')
    with open('reddit.csv', 'w') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        for submission in resultsList:
            for comment in submission.comments.list():
                comm = comment.body.encode()
                fileWriter.writerow([n,submission.id,submission.title,comm,comment.score,query,comment.subreddit.name,comment.id])



if __name__ == '__main__':
    mineComments((getSubmissions(query)))