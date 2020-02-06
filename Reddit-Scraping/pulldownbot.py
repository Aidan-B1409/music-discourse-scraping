import praw
import csv
from datetime import datetime

reddit = praw.Reddit('bot1')
"""TODO - Make megasubreddit object based on subreddits.txt"""
subreddit = reddit.subreddit("all")
"""TODO - Automate searching of multiple queries based on txt object"""
query = 'title:"Madeon" "All My Friends"'
"""Very important constant, defines how deep the getSubmissions method will search"""
SEARCHLIMIT = 3

def getSubmissions(keyword):
    searchResults = list()
    for submission in subreddit.search(keyword, 'top', 'lucene', "all",limit=SEARCHLIMIT):
        submission.comments.replace_more()
        searchResults.append(submission)
    return searchResults
def mineComments(resultsList):
    """counter for CSV rows"""
    n = 0
    """Time of CSV initialization"""
    dtime_string = datetime.now().strftime('%d-%m-%Y%H-%M-%S')
    with open('reddit_'+dtime_string+'.csv', 'w', encoding='utf-8', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fileWriter.writerow(["Query Index", "Query", "Submission Index", "Subreddit", "Submission Title", "Submission ID", "Submission Score", "Comment", "Comment ID", "Comment Index", "Comment Score" ])

        for submission in resultsList:
            for comment in submission.comments:
                submissionID = submission.id
                submissionTitle = submission.title
                commentBody = comment.body.replace(",","").encode('utf-8', "ignore");
                commentBody = str(commentBody, encoding = 'utf-8');
                commentScore = comment.score
                entryQuery = query
                subredditName = comment.subreddit.display_name
                print(subredditName)
                commentID = comment.id

                fileWriter.writerow([n, submissionID, submissionTitle, commentBody, commentScore, entryQuery, subredditName, commentID])
                n = n + 1



if __name__ == '__main__':
    mineComments((getSubmissions(query)))