import pandas as pd
import csv
import os
from os import walk
from datetime import datetime

from EmoVAD_wlist import EmoVAD_wlist
from FeatureGenerator import FeatureGenerator
from analysis import analyze_features


# Some terminology to help out whoever has to interpret this 
# song_df -- The raw data, grabbed from the director
# wordlist_df -- The dataframe holding the semantic wordlist
# word_df -- A wordlist which holds each unique comment in the glob/comment and its # of occurances 
# glob -- The concatanation 
class App:

    # NOTE - If you're getting "dict contains fields not in fieldnames" errors, it is probably because 
    # you added features/keys within one of the wlist classes without also adding it here
    # I know it's a gross solution but this is the only way to generate the header and lazy load the features. 
    # In the future - please replace this key system with a JSON. 
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

        "EmoVAD_glob_v_sqmean": 0.0, "EmoVAD_glob_a_sqmean": 0.0, 
        "EmoVAD_glob_d_sqmean": 0.0, "EmoVAD_glob_v_sqmean_uniq": 0.0, 
        "EmoVAD_glob_a_sqmean_uniq": 0.0, "EmoVAD_glob_d_sqmean_uniq": 0.0, 

        "EmoVAD_glob_max_word_v": 0.0, "EmoVAD_glob_max_word_a": 0.0,
        "EmoVAD_glob_max_word_d": 0.0, 
        
        "EmoVAD_glob_min_word_v": 0.0,
        "EmoVAD_glob_min_word_a": 0.0, "EmoVAD_glob_min_word_d": 0.0,

        "EmoVAD_glob_most_word_v": 0.0, "EmoVAD_glob_most_word_a": 0.0,
        "EmoVAD_glob_most_word_d": 0.0, "EmoVAD_glob_most_word_count": 0.0,

        "EmoVAD_glob_v_ratio": 0.0, "EmoVAD_glob_a_ratio": 0.0, 
        "EmoVAD_glob_d_ratio": 0.0, 
        "EmoVAD_glob_v_uniq_ratio": 0.0, "EmoVAD_glob_a_uniq_ratio": 0.0, 
        "EmoVAD_glob_d_uniq_ratio": 0.0, 

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

        "EmoVAD_max_word_v_sqmean": 0.0, "EmoVAD_max_word_v_sqstd": 0.0,
        "EmoVAD_max_word_a_sqmean": 0.0, "EmoVAD_max_word_a_sqstd": 0.0,
        "EmoVAD_max_word_d_sqmean": 0.0, "EmoVAD_max_word_d_sqstd": 0.0,
        
        "EmoVAD_min_word_v_mean": 0.0, "EmoVAD_min_word_v_std": 0.0,
        "EmoVAD_min_word_a_mean": 0.0, "EmoVAD_min_word_a_std": 0.0,
        "EmoVAD_min_word_d_mean": 0.0, "EmoVAD_min_word_d_std": 0.0,

        "EmoVAD_most_word_v_mean": 0.0, "EmoVAD_most_word_v_std": 0.0,
        "EmoVAD_most_word_a_mean": 0.0, "EmoVAD_most_word_a_std": 0.0,
        "EmoVAD_most_word_d_mean": 0.0, "EmoVAD_most_word_d_std": 0.0,
        "EmoVAD_most_word_count_mean": 0.0, "EmoVAD_most_word_count_std": 0.0,


        "bsmVAD_glob_v_mean": 0.0, "bsmVAD_glob_v_stdev": 0.0,
        "bsmVAD_glob_a_mean": 0.0, "bsmVAD_glob_a_stdev": 0.0,
        "bsmVAD_glob_d_mean": 0.0, "bsmVAD_glob_d_stdev": 0.0,
        "bsmVAD_glob_v_mean_uniq": 0.0, "bsmVAD_glob_v_stdev_uniq": 0.0,
        "bsmVAD_glob_a_mean_uniq": 0.0, "bsmVAD_glob_a_stdev_uniq": 0.0,
        "bsmVAD_glob_d_mean_uniq": 0.0, "bsmVAD_glob_d_stdev_uniq": 0.0,

        "bsmVAD_glob_v_sqmean": 0.0, "bsmVAD_glob_a_sqmean": 0.0, 
        "bsmVAD_glob_d_sqmean": 0.0, "bsmVAD_glob_v_sqmean_uniq": 0.0, 
        "bsmVAD_glob_a_sqmean_uniq": 0.0, "bsmVAD_glob_d_sqmean_uniq": 0.0, 

        "bsmVAD_glob_max_word_v": 0.0, "bsmVAD_glob_max_word_a": 0.0,
        "bsmVAD_glob_max_word_d": 0.0, 
        
        "bsmVAD_glob_min_word_v": 0.0,
        "bsmVAD_glob_min_word_a": 0.0, "bsmVAD_glob_min_word_d": 0.0,

        "bsmVAD_glob_most_word_v": 0.0, "bsmVAD_glob_most_word_a": 0.0,
        "bsmVAD_glob_most_word_d": 0.0, "bsmVAD_glob_most_word_count": 0.0,

        "bsmVAD_glob_v_ratio": 0.0, "bsmVAD_glob_a_ratio": 0.0, 
        "bsmVAD_glob_d_ratio": 0.0, 
        "bsmVAD_glob_v_uniq_ratio": 0.0, "bsmVAD_glob_a_uniq_ratio": 0.0, 
        "bsmVAD_glob_d_uniq_ratio": 0.0, 

        "bsmVAD_v_mean_mean": 0.0, "bsmVAD_v_mean_stdev": 0.0,
        "bsmVAD_v_std_mean": 0.0, "bsmVAD_v_stdev_stdev": 0.0,
        "bsmVAD_v_uniq_mean_mean": 0.0, "bsmVAD_v_uniq_mean_stdev": 0.0,
        "bsmVAD_v_uniq_std_mean": 0.0, "bsmVAD_v_uniq_stdev_stdev": 0.0,

        "bsmVAD_a_mean_mean": 0.0, "bsmVAD_a_mean_stdev": 0.0,
        "bsmVAD_a_std_mean": 0.0, "bsmVAD_a_stdev_stdev": 0.0,
        "bsmVAD_a_uniq_mean_mean": 0.0, "bsmVAD_a_uniq_mean_stdev": 0.0,
        "bsmVAD_a_uniq_std_mean": 0.0, "bsmVAD_a_uniq_stdev_stdev": 0.0,

        "bsmVAD_d_mean_mean": 0.0, "bsmVAD_d_mean_stdev": 0.0,
        "bsmVAD_d_std_mean": 0.0, "bsmVAD_d_stdev_stdev": 0.0,
        "bsmVAD_d_uniq_mean_mean": 0.0, "bsmVAD_d_uniq_mean_stdev": 0.0,
        "bsmVAD_d_uniq_std_mean": 0.0, "bsmVAD_d_uniq_stdev_stdev": 0.0,

        "bsmVAD_max_word_v_mean": 0.0, "bsmVAD_max_word_v_std": 0.0,
        "bsmVAD_max_word_a_mean": 0.0, "bsmVAD_max_word_a_std": 0.0,
        "bsmVAD_max_word_d_mean": 0.0, "bsmVAD_max_word_d_std": 0.0,

        "bsmVAD_max_word_v_sqmean": 0.0, "bsmVAD_max_word_v_sqstd": 0.0,
        "bsmVAD_max_word_a_sqmean": 0.0, "bsmVAD_max_word_a_sqstd": 0.0,
        "bsmVAD_max_word_d_sqmean": 0.0, "bsmVAD_max_word_d_sqstd": 0.0,
        
        "bsmVAD_min_word_v_mean": 0.0, "bsmVAD_min_word_v_std": 0.0,
        "bsmVAD_min_word_a_mean": 0.0, "bsmVAD_min_word_a_std": 0.0,
        "bsmVAD_min_word_d_mean": 0.0, "bsmVAD_min_word_d_std": 0.0,

        "bsmVAD_most_word_v_mean": 0.0, "bsmVAD_most_word_v_std": 0.0,
        "bsmVAD_most_word_a_mean": 0.0, "bsmVAD_most_word_a_std": 0.0,
        "bsmVAD_most_word_d_mean": 0.0, "bsmVAD_most_word_d_std": 0.0,
        "bsmVAD_most_word_count_mean": 0.0, "bsmVAD_most_word_count_std": 0.0,
    }
    
    def __init__(self,comment_path = "") -> None:
        self.comment_path = comment_path

    def song_csv_generator(self):
        for subdir, dirs, files in walk(self.comment_path):
            for file in files:
                fdir = subdir + "/" + file
                yield pd.read_csv(fdir, encoding="utf-8", index_col = False, engine="c")

    def main(self) -> None:
        self._find_intersects()
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


    # GARBAGE UTILITY METHOD FOR TESTING INTERSECTIONS BETWEEN WORDLISTS - DELETEME
    def _find_intersects(self):
        bsmvad_df = pd.read_csv(os.getcwd() + '/wordlists/BRM-emot-submit.csv', encoding='utf-8', engine='python')
        # drop unneeded columns
        bsmvad_df.drop(bsmvad_df.iloc[:, 10:64].columns, axis = 1, inplace = True) 
        bsmvad_df.drop(['V.Rat.Sum', 'A.Rat.Sum','D.Rat.Sum'], axis = 1, inplace = True) 
        # drop blank rows, if any
        bsmvad_df = bsmvad_df[bsmvad_df['Word'].notnull()]

        #open emovad
        emovad_df = pd.read_csv(os.getcwd() + '/wordlists/NRC-VAD-Lexicon.txt', names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t')

        print("EmoVAD U BsmVAD: " + str(len(set(emovad_df['Word']) & set(bsmvad_df['Word']))))   

        #open emolex
        emolex_df = pd.read_csv(os.getcwd() + '/wordlists/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',  names=['Word','Emotion','Association'], skiprows=1, sep='\t') 
        print("EmoVAD U EmoLEX: " + str(len(set(emovad_df['Word']) & set(emolex_df['Word']))))  


if __name__ == "__main__":
    fe = App(comment_path="/mnt/g/new_data/subset_deezer_test")
    fe.main()
