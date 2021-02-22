from bsmvad_wlist import BSMVAD_wlist
import pandas as pd
import csv
import os
from os import walk
from datetime import datetime

import EmoVAD_wlist
import bsmvad_wlist
import mpqa_wlist
import emolex_wlist
import emoaff_wlist
from FeatureGenerator import FeatureGenerator
from analysis import analyze_features


# Some terminology to help out whoever has to interpret this 
# song_df -- The raw data, grabbed from the director
# wordlist_df -- The dataframe holding the semantic wordlist
# word_df -- A wordlist which holds each unique comment in the glob/comment and its # of occurances 
# glob -- The concatanation 
class App:

    m_features = {'Song_ID': "", 'Song_Name': "", 'n_words': -1, 'comment_length_stdev': -1, 'Song_Artist': "",
     'existing_valence': "", 'existing_arousal': "", 'n_words_uniq': -1, 'n_comments': -1, 'comment_length_mean': -1}
    wordlists = {EmoVAD_wlist, bsmvad_wlist, mpqa_wlist, emolex_wlist, emoaff_wlist}
    for wlist in wordlists:
        m_features.update(wlist.get_header())
    
    def __init__(self,comment_path = "") -> None:
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
                features = FeatureGenerator(song_df).get_features()
                csvwriter.writerow(features)

        # after we finish generating all our features - do some simple analysis
        analysis_csv_name = "feature_analysis" + timestamp + ".csv"
        analyze_features(data_csv_name, analysis_csv_name, self.m_features)

if __name__ == "__main__":
    fe = App(comment_path="/mnt/g/new_data/subset_deezer_test")
    fe.main()
