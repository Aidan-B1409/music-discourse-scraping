import praw

reddit = praw.Reddit('bot1')
subreddit = reddit.subreddit("music")

for submission in subreddit.hot(limit=5):
    print("Title: ", submission.title)