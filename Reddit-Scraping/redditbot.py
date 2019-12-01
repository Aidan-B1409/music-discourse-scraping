import praw
import csv
from datetime import datetime

reddit = praw.Reddit('bot1')
subreddit = reddit.subreddit("music")

def MineSubreddit(sub):
    n = 0
    dtime_string = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    for submission in sub.hot(limit=1):
        print("Title: ", submission.title)
        submission.comments.replace_more(limit=None)
        with open('data' + dtime_string + '.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|')
            for comment in submission.comments.list():
                comm = comment.body.encode()
                filewriter.writerow([n, comm])
                n += 1


def searchSubreddit(sub, keyword):
    #Creates a search query which is sorted by relevance to keyword, syntax = lucene, time-filter = all
    for submission in sub.search(keyword, 'relevance', 'lucene', 'all'):
        print(submission.title)

def getSubreddits(data_txt):
    #read subreddit data file
    with open(data_txt) as f:
        content = f.readlines()
    #strip out newline characters
    content = [x.replace('\n','+') for x in content]
    return content

if __name__ == '__main__':
    MineSubreddit(subreddit)
