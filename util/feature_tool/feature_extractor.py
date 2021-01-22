import pandas as pd
import re
from os import walk
from datetime import datetime

from wordlists import WordLists
from csv_builder import CSVBuilder


class FeatureExtractor:
    
    def __init__(self, wordlists = WordLists(), comment_path = "") -> None:
        self.wordlists = wordlists
        self.comment_path = comment_path

    def song_csv_generator(self):
        for subdir, dirs, files in walk(self.comment_path):
            for file in files:
                fdir = subdir + "/" + file
                yield pd.read_csv(fdir, encoding="utf-8", index_col = False, engine="c")

    def main(self) -> None:
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        csv_name = "deezer_features_" + timestamp + ".csv"
        with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            csv = CSVBuilder(csvfile)
            for df in self.song_csv_generator():
                csv.writerow(self.generate_features(df))

    # Generate song level features
    def generate_features(self, df) -> list:
        features = []
        features.append(str(self.song_id(df)))
        features.append(self.song_name(df))        
        features.append(self.n_comments(df))
        return features

    #I want this as an int unlike the other features so that I can use it in following calculations. 
    def n_comments(self, df) -> int:
        #We subtract one from the length because the last row is just metadata, not comments. 
        return len(df) - 1

    def song_id(self, df) -> str:
        return str(df.iloc[0]['Song ID'])

    def song_name(self, df) -> str:
        query = str(df.iloc[0]['Query'])
        songname = re.sub(r"title:", "", query)
        songname = re.sub(r'"', "", songname)
        return songname


if __name__ == "__main__":
    fe = FeatureExtractor(comment_path="/mnt/g/new_data/subset_deezer_test")
    fe.main()

