import praw

reddit = praw.Reddit('bot1')
subreddit = reddit.subreddit("music")

def MineSubreddit(sub):
    n = 0
    for submission in sub.hot(limit=1):
        print("Title: ", submission.title)
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            print(comment.body)
            n = n + 1
        print(n)


def SearchSubreddit(sub, keyword):
    #Creates a search query which is sorted by relevance to keyword, syntax = lucene, time-filter = all
    for submission in sub.search(keyword, 'relevance', 'lucene', 'all'):
        print(submission.title)

def getSubreddits(data_txt):
    #read subreddit data file
    with open(data_txt) as f:
        content = f.readlines()
    #strip out newline characters
    content = [x.strip('\n') for x in content]


if __name__ == '__main__':
    getSubreddits(r'subreddits.txt')
