import praw
from praw.models import MoreComments

reddit = praw.Reddit('bot1')
subreddit = reddit.subreddit("music")
n = 0
for submission in subreddit.hot(limit=1):
    print("Title: ", submission.title)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        print(comment.body)
        n = n + 1
    print(n)
