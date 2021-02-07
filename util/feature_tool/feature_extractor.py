import pandas as pd
import csv
from os import walk
from datetime import datetime

from wordlists import WordLists
from glob_maker import make_glob
from meta_generator import MetaGenerator
from EmoVAD_generator import EmoVADGenerator
from analysis import analyze_features


# Some terminology to help out whoever has to interpret this 
# song_df -- The raw data, grabbed from the director
# wordlist_df -- The dataframe holding the semantic wordlist
# word_df -- A wordlist which holds each unique comment in the glob/comment and its # of occurances 
# glob -- The concatanation 
class FeatureExtractor:

    m_features = {
        "Song_ID": "",
        "Song_Artist": "", "Song_Name": "",
        "existing_valence": "", "existing_arousal": "",
        "n_comments": -1, "n_words": -1, "n_words_uniq": -1,
        "comment_length_mean": 0, "comment_length_stdev": 0,
        
        "EmoVAD_glob_v_mean": 0.0, "EmoVAD_glob_v_stdev": 0.0,
        "EmoVAD_glob_a_mean": 0.0, "EmoVAD_glob_a_stdev": 0.0,
        "EmoVAD_glob_d_mean": 0.0, "EmoVAD_glob_d_stdev": 0.0,
        "EmoVAD_glob_v_mean_uniq": 0.0, "EmoVAD_glob_v_stdev_uniq": 0.0,
        "EmoVAD_glob_a_mean_uniq": 0.0, "EmoVAD_glob_a_stdev_uniq": 0.0,
        "EmoVAD_glob_d_mean_uniq": 0.0, "EmoVAD_glob_d_stdev_uniq": 0.0,

        "EmoVAD_glob_max_word_v": 0.0, "EmoVAD_glob_max_word_a": 0.0,
        "EmoVAD_glob_max_word_d": 0.0, 
        
        "EmoVAD_glob_min_word_v": 0.0,
        "EmoVAD_glob_min_word_a": 0.0, "EmoVAD_glob_min_word_d": 0.0,

        "EmoVAD_glob_most_word_v": 0.0, "EmoVAD_glob_most_word_a": 0.0,
        "EmoVAD_glob_most_word_d": 0.0, "EmoVAD_glob_most_word_count": 0.0,

        "EmoVAD_v_mean_mean": 0.0, "EmoVAD_v_mean_stdev": 0.0,
        "EmoVAD_v_std_mean": 0.0, "EmoVAD_v_stdev_stdev": 0.0,
        "EmoVAD_v_uniq_mean_mean": 0.0, "EmoVAD_v_uniq_mean_stdev": 0.0,
        "EmoVAD_v_uniq_std_mean": 0.0, "EmoVAD_v_uniq_stdev_stdev": 0.0,

        "EmoVAD_a_mean_mean": 0.0, "EmoVAD_a_mean_stdev": 0.0,
        "EmoVAD_a_std_mean": 0.0, "EmoVAD_a_stdev_stdev": 0.0,
        "EmoVAD_a_uniq_mean_mean": 0.0, "EmoVAD_a_uniq_mean_stdev": 0.0,
        "EmoVAD_a_uniq_std_mean": 0.0, "EmoVAD_a_uniq_stdev_stdev": 0.0,

        "EmoVAD_d_mean_mean": 0.0, "EmoVAD_d_mean_stdev": 0.0,
        "EmoVAD_d_std_mean": 0.0, "EmoVAD_d_stdev_stdev": 0.0,
        "EmoVAD_d_uniq_mean_mean": 0.0, "EmoVAD_d_uniq_mean_stdev": 0.0,
        "EmoVAD_d_uniq_std_mean": 0.0, "EmoVAD_d_uniq_stdev_stdev": 0.0,

        "EmoVAD_max_word_v_mean": 0.0, "EmoVAD_max_word_v_std": 0.0,
        "EmoVAD_max_word_a_mean": 0.0, "EmoVAD_max_word_a_std": 0.0,
        "EmoVAD_max_word_d_mean": 0.0, "EmoVAD_max_word_d_std": 0.0,
        
        "EmoVAD_min_word_v_mean": 0.0, "EmoVAD_min_word_v_std": 0.0,
        "EmoVAD_min_word_a_mean": 0.0, "EmoVAD_min_word_a_std": 0.0,
        "EmoVAD_min_word_d_mean": 0.0, "EmoVAD_min_word_d_std": 0.0,

        "EmoVAD_most_word_v_mean": 0.0, "EmoVAD_most_word_v_std": 0.0,
        "EmoVAD_most_word_a_mean": 0.0, "EmoVAD_most_word_a_std": 0.0,
        "EmoVAD_most_word_d_mean": 0.0, "EmoVAD_most_word_d_std": 0.0,
        "EmoVAD_most_word_count_mean": 0.0, "EmoVAD_most_word_count_std": 0.0,

    }
    
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
        data_csv_name = "deezer_features_" + timestamp + ".csv"
        with open(data_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.DictWriter(csvfile, self.m_features.keys())
            csvwriter.writeheader()
            for song_df in self.song_csv_generator():
                features = {}
                glob_df = make_glob(song_df)
                features = {**features, **MetaGenerator(song_df, glob_df).get_features()}
                features = {**features, **EmoVADGenerator(song_df, glob_df, self.wordlists).get_features()}
                csvwriter.writerow(features)

        # after we finish generating all our features - do some simple analysis
        analysis_csv_name = "feature_analysis" + timestamp + ".csv"
        analyze_features(data_csv_name, analysis_csv_name, self.m_features)
            


if __name__ == "__main__":
    fe = FeatureExtractor(comment_path="/mnt/g/new_data/subset_deezer_test")
    fe.main()