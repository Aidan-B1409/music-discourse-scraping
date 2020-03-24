import praw
import sys, getopt
import pandas as pd
from query import Query

reddit = praw.Reddit('bot1')

def getArgs(argv):
    inputFile = ""
    try:
        opts, args = getopt.getopt(argv, "hi:",["inputFile="])
    except getopt.GetoptError:
        print("Invalid File Arguments Supplied! \n Valid Arguments Include: \n "
              " 1: h (Help) \n 2: -i (Input CSV File) (Also accepts inputFile=)")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
           print("Valid Arguments Include: \n 1: h (Help) \n  2: -i (Input CSV File) (Also accepts inputFile=)")
           sys.exit(2)
        elif opt in ("-i", "--inputFile"):
            inputFile = arg
    return inputFile
def readQueries(inputFile):
    df = pd.read_csv(inputFile.strip())
    for index, row in df.iterrows():
        print(row['artist_name'], row['track_name'])
        query = Query(row['artist_name'], row['track_name'], reddit)
        valence = row['valence']
        arousal = row['arousal']
        deezerID = row['dzr_sng_id']
        query.mineComments(index, valence, arousal, deezerID)


if __name__ == '__main__':
    inputFile = getArgs(sys.argv[1:])
    readQueries(inputFile)
    #MANUAL TEST QUERY
    #query = Query("Madeon", "All My Friends", reddit)
    #query.mineComments("0", "0", "0", "0")
