import praw
from praw.models import MoreComments

reddit = praw.Reddit('bot1')
subreddit = reddit.subreddit("music")

for submission in subreddit.hot(limit=1):
    print("Title: ", submission.title)
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue
        print(top_level_comment.body)