import csv
import datetime
from tqdm import tqdm
import time
from datetime import datetime


class Query:
    query = "null"
    SEARCHLIMIT = 3;

    def __init__(self, track, artist, reddit):
        self.track = track
        self.artist = artist
        self.reddit = reddit
        self.subreddit = reddit.subreddit("all")
        self.query = self.buildQuery(track, artist)

    def buildQuery(self, track, artist):
        return str("title:" + "\"" + track + "\"" + " " + "\"" + artist + "\"")

    def getSubmissions(self):
        submissionList = list()
        subreddit = self.reddit.subreddit("all")
        for submission in subreddit.search(self.query, 'top', 'lucene', "all", limit=self.SEARCHLIMIT):
            submission.comments.replace_more()
            submissionList.append(submission)
        return submissionList

    def mineComments(self, queryIndex, valence, arousal, deezerID):
        """Time of CSV initialization"""
        dtime_string = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        fileName = 'reddit_' + dtime_string + "_" + str(deezerID) + ".csv"
        with open(fileName, 'w', newline='', encoding='utf-8') as csvfile:
            fileWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fileWriter.writerow(["Query Index", "Query", "Valence", "Arousal", "Result Index", "Subreddit", "Subreddit ID",
                                 "Submission Title", "Submission Body", "SubmissionID", "Comment Body", "Comment ID", "Comment Index", "Comment Replies",
                                 "Comment Score", "Submission Comments", "Submission URL", "Submission Score", "Deezer ID"])
            submissionsList = self.getSubmissions()
            if len(submissionsList) > 0:
                for resultIndex, submission in enumerate(tqdm(submissionsList)):
                    for commentIndex, comment in enumerate(submission.comments):
                        submissionID = submission.id
                        submissionTitle = submission.title
                        submissionBody = submission.selftext
                        commentBody = comment.body
                        commentScore = comment.score
                        commentID = comment.id
                        subredditName = comment.subreddit.display_name
                        subredditID = comment.subreddit.id
                        commentID = comment.id
                        commentReplies = len(comment.replies)
                        submissionComments = submission.num_comments
                        submissionURL = submission.url
                        submissionScore = submission.score
                        if commentBody.strip() != "[deleted]":
                            fileWriter.writerow(
                                [queryIndex, self.query, valence, arousal, resultIndex, subredditName, subredditID,
                                 submissionTitle, submissionBody, submissionID, commentBody, commentID, commentIndex, commentReplies,
                                 commentScore, submissionComments, submissionURL, submissionScore, deezerID])
                    time.sleep(0.1)
                else:
                    fileWriter.writerow(
                        [queryIndex, self.query, valence, arousal, "", "", "", "",
                         "", "", "End Of File", "", "", "", "",
                         "", "", "", deezerID])
            else:
                fileWriter.writerow(
                    [queryIndex, self.query, valence, arousal, "", "", "", "",
                     "", "", "No Results", "", "", "", "",
                     "", "", "", deezerID])
                print("No Results")
