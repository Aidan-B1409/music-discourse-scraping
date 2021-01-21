from os import walk
import pandas as pd
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
        features.append(str(self.n_comments(df)))
        return features

    def n_comments(self, df):
        return len(df)

    # def song_id(self, df):
    #     return df[deezer]


if __name__ == "__main__":
    fe = FeatureExtractor(comment_path="/mnt/d/Datasets/deezer_new")
    fe.main()

