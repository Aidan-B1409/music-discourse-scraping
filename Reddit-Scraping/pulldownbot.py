import praw
import sys
import getopt
import pandas as pd
from query import Query

reddit = praw.Reddit('bot1')


def get_args(argv: str):
    input_file = ""
    try:
        opts, args = getopt.getopt(argv, "hi:", ["input_file="])
    except getopt.GetoptError:
        print("Invalid File Arguments Supplied! \n Valid Arguments Include: \n "
              " 1: h (Help) \n 2: -i (Input CSV File) (Also accepts input_file=)")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Valid Arguments Include: \n 1: h (Help) \n  2: -i (Input CSV File) (Also accepts input_file=)")
            sys.exit(2)
        elif opt in ("-i", "--input_file"):
            input_file = arg
    return input_file


def read_queries(input_file: str) -> None:
    df = pd.read_csv(input_file.strip())
    for index, row in df.iterrows():
        print(row['artist_name'], row['track_name'])
        query = Query(row['artist_name'], row['track_name'], reddit)
        valence = row['valence']
        arousal = row['arousal']
        deezer_id = row['dzr_sng_id']
        query.mine_comments(index, valence, arousal, deezer_id)


if __name__ == '__main__':
    inputFile = get_args(sys.argv[1:])
    read_queries(inputFile)
    # MANUAL TEST QUERY
    # search_keyword = Query("Madeon", "All My Friends", reddit)
    # search_keyword.mineComments("0", "0", "0", "0")
