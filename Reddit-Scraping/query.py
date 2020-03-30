import csv
import datetime
import time
from tqdm import tqdm
from datetime import datetime


def build_query(track: str, artist: str) -> str:
    return str("title:" + "\"" + track + "\"" + " " + "\"" + artist + "\"")


class Query:
    search_keyword = "null"
    SEARCH_LIMIT = 3

    def __init__(self, cwd: str, track: str, artist: str, reddit: object):
        self.cwd = cwd
        self.track = track
        self.artist = artist
        self.reddit = reddit
        self.subreddit = reddit.subreddit("all")
        self.search_keyword = build_query(track, artist)

    def get_submissions(self) -> list:
        submission_list = list()
        subreddit = self.reddit.subreddit("all")
        for submission in subreddit.search(self.search_keyword, 'top', 'lucene', "all", limit=self.SEARCH_LIMIT):
            submission.comments.replace_more()
            submission_list.append(submission)
        return submission_list

    def mine_comments(self, query_index: str, valence: str, arousal: str, deezer_id: str) -> None:
        """Time of CSV initialization"""
        dtime_string = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        file_name = 'reddit_' + dtime_string + "_" + str(deezer_id) + ".csv"
        with open(self.cwd + file_name, 'w', newline='', encoding='utf-8') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(
                ["Query Index", "Query", "Valence", "Arousal", "Result Index", "Subreddit", "Subreddit ID",
                 "Submission Title", "Submission Body", "SubmissionID", "Comment Body", "Comment ID", "Comment Index",
                 "Comment Replies", "Comment Score", "Submission Comments", "Submission URL", "Submission Score"
                 "Deezer ID"])
            submissions_list = self.get_submissions()
            if len(submissions_list) > 0:
                for result_index, submission in enumerate(tqdm(submissions_list)):
                    for comment_index, comment in enumerate(submission.comments):
                        submission_id = submission.id
                        submission_title = submission.title
                        submission_body = submission.selftext
                        comment_body = comment.body
                        comment_score = comment.score
                        subreddit_name = comment.subreddit.display_name
                        subreddit_id = comment.subreddit.id
                        comment_id = comment.id
                        comment_replies = len(comment.replies)
                        submission_comments = submission.num_comments
                        submission_url = submission.url
                        submission_score = submission.score
                        if comment_body.strip() != "[deleted]":
                            file_writer.writerow(
                                [query_index, self.search_keyword, valence, arousal, result_index, subreddit_name,
                                 subreddit_id, submission_title, submission_body, submission_id, comment_body,
                                 comment_id, comment_index, comment_replies, comment_score, submission_comments,
                                 submission_url, submission_score, deezer_id])
                    time.sleep(0.1)
                else:
                    file_writer.writerow(
                        [query_index, self.search_keyword, valence, arousal, "", "", "", "",
                         "", "", "End Of File", "", "", "", "",
                         "", "", "", deezer_id])
            else:
                file_writer.writerow(
                    [query_index, self.search_keyword, valence, arousal, "", "", "", "",
                     "", "", "No Results", "", "", "", "",
                     "", "", "", deezer_id])
                print("No Results")
