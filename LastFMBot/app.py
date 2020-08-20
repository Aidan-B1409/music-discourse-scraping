import pandas as pd
import pylast
from pylast import LastFMNetwork
import configparser
import os
import sys
import getopt
from query import Query

CONFIG_PATH = "config.ini"
API_KEY = "invalid"
API_SECRET = "invalid"
USERNAME = "invalid"
INPUT_FILE = ""
cwd = os.path.join(os.getcwd(), "downloads/")


def get_config(config_path: str) -> None:
    config = configparser.ConfigParser()
    config.read(config_path)
    global API_KEY, API_SECRET, USERNAME
    API_KEY = config['LASTFM_API']['API_KEY']
    API_SECRET = config['LASTFM_API']['API_SECRET']
    USERNAME = config['LASTFM_API']['CLIENT_NAME']

def get_args(argv: str) -> None:
    global INPUT_FILE
    try:
        opts, args = getopt.getopt(argv, "hi:", ["INPUT_FILE="])
    except getopt.GetoptError:
        print("Invalid File Arguments Supplied! \n Valid Arguments Include: \n "
                " 1: h (Help) \n 2: -i (Input CSV File) (Also accepts input_file=)")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Valid Arguments Include: \n 1: h (Help) \n  2: -i (Input CSV File) (Also accepts input_file=)")
            sys.exit(2)
        elif opt in ("-i", "--input_file"):
            
            INPUT_FILE = arg

def read_queries(input_file: str, network: LastFMNetwork) -> None:
    df = pd.read_csv(input_file.strip())
    for index, row in df.iterrows():
        artist = row['artist_name'].strip()
        song = row['track_name'].strip()
        query = Query(cwd, song, artist, network)
        valence = row['valence']
        arousal = row['arousal']
        song_id = row['dzr_sng_id']
        print(artist, song)
        query.mine_tags(index, valence, arousal, song_id)

def make_downloads() -> None:
    try:
        os.mkdir(cwd)
    except OSError as e:
        print(e)
        return

def main():
    get_config(CONFIG_PATH)
    get_args(sys.argv[1:])
    make_downloads()
    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET, username=USERNAME)
    read_queries(INPUT_FILE, network)



if __name__ == '__main__':

    main()

