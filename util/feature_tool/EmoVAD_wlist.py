import pandas as pd 

from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

def get_glob_headers() -> dict:
    dict = {}
    for datatype in ['v', 'a', 'd']:
            for analysis_type in ['mean', 'std', 'ratio']:
                for uniq_status in ['uniq_', '']:
                    dict[f"EmoVAD_glob_{datatype}_{uniq_status}{analysis_type}"] = 0.0
    for datatype in ['v', 'a', 'd']:
        for analysis_type in ['max_word', 'min_word', 'most_word']:
            dict[f"EmoVAD_glob_{datatype}_{analysis_type}"] = 0.0
    return dict

def get_commentlevel_headers() -> dict:
    dict = {}
    for datatype in ['v', 'a', 'd']:
        for analysis_type in ['means', 'stds']:
            for uniq_status in ['_uniq', '']:
                for value in ['mean', 'std']:
                    dict[f'EmoVAD_{datatype}{uniq_status}_{analysis_type}_{value}'] = 0.0
    for datatype in ['v', 'a', 'd']:
        for analysis_type in ['max_words', 'min_words', 'most_words']:
            for value in ['mean', 'std']:
                dict[f'EmoVAD_{datatype}_{analysis_type}_{value}'] = 0.0
    return dict

def get_header():
    header = {}
    header.update(get_glob_headers())
    header.update(get_commentlevel_headers())
    return header
    

class EmoVAD_wlist:

    def __init__(self, emovad_df) -> None:

        columns = []
        for datatype in ['v', 'a', 'd']:
            for analysis in ['means', 'stds']:
                for uniq_status in ['_uniq', '']:
                    columns.append(f"{datatype}{uniq_status}_{analysis}")
        for datatype in ['v', 'a', 'd']:
            for analysis in ['max_words', 'min_words', 'most_words']:
                columns.append(f"{datatype}_{analysis}")

        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.emovad_df = emovad_df
        self.features_wordlevel = get_glob_headers()
        self.features_commentlevel = get_commentlevel_headers()

    def wordlevel_analysis(self, song_df, glob_df) -> dict:

        if(len(song_df) > 0):
            matched_word_df = unsquished_intersection(song_df, self.emovad_df)
            uniq_matched_word_df = glob_intersection(glob_df, self.emovad_df)

            if(len(uniq_matched_word_df) > 0):
                root = "EmoVAD_glob"
                # where data_type is assosciated to the feature dict built for the class
                # and data_key is assosciated to the wordlist dataframe
                # e.g. data_type = 'v', data_key = 'Valence'
                for data_type, data_key in self._itervad():
                    self._write_mean_std(self.features_wordlevel, f"{root}_{data_type}", data_key, matched_word_df)
                    
                    ratio_threshold = self.features_wordlevel[f"{root}_{data_type}_mean"]
                    self._write_ratio(self.features_wordlevel, f"{root}_{data_type}", data_key, ratio_threshold, matched_word_df)
                    
                    self._write_mean_std(self.features_wordlevel, f"{root}_{data_type}_uniq", data_key, uniq_matched_word_df)
                    # unique ratios
                    uniq_ratio_threshold = self.features_wordlevel[f"{root}_{data_type}_uniq_mean"]
                    
                    self._write_ratio(self.features_wordlevel, f"{root}_{data_type}_uniq", data_key, uniq_ratio_threshold, uniq_matched_word_df)
                    self._write_min(self.features_wordlevel, uniq_matched_word_df, f"{root}_{data_type}", data_key)
                    self._write_max(self.features_wordlevel, uniq_matched_word_df, f"{root}_{data_type}", data_key)
                    self._write_most(self.features_wordlevel, uniq_matched_word_df, f"{root}_{data_type}", data_key)

        return self.features_wordlevel

    def process_comment(self, index, comment):
        comment_list = clean_comment(comment)

        unique_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])

        found_uniq_word_df = glob_intersection(unique_words_df, self.emovad_df)
        found_words_df = pd.merge(words_df, self.emovad_df, on='Word')

        if len(found_uniq_word_df) > 0:
            for data_type, data_key in self._itervad():
                # get_mean_std returns a (mean, std) tuple
                data = get_mean_std(found_words_df[data_key], len(found_words_df))
                self.comment_analysis_df.at[index, f"{data_type}_means"] = data[0]
                self.comment_analysis_df.at[index, f"{data_type}_stds"] = data[1]

                uniq_data = get_mean_std(found_uniq_word_df[data_key], len(found_uniq_word_df))
                self.comment_analysis_df.at[index, f"{data_type}_uniq_means"] = uniq_data[0]
                self.comment_analysis_df.at[index, f"{data_type}_uniq_stds"] = uniq_data[1]

                minmaxmost = self._get_minmaxmost_words(found_uniq_word_df, data_key)
                self.comment_analysis_df.at[index, f"{data_type}_max_words"] = minmaxmost['max_word_score']
                self.comment_analysis_df.at[index, f"{data_type}_min_words"] = minmaxmost['min_word_score']
                self.comment_analysis_df.at[index, f"{data_type}_most_words"] = minmaxmost['most_word_score']

    def analyze_comments(self) -> dict:
        root = "EmoVAD"
        for data_type, data_key in self._itervad():
            for analysis_type in ['means', 'stds']:
                for uniq_status in ['_uniq', '']:
                    feature_key = f"{root}_{data_type}{uniq_status}_{analysis_type}"
                    df_key = f"{data_type}{uniq_status}_{analysis_type}"
                    self._write_mean_std(self.features_commentlevel, feature_key, df_key, self.comment_analysis_df)
        for data_type, data_key in self._itervad():
            for analysis_type in ['max_words', 'min_words', 'most_words']:
                feature_key = f"{root}_{data_type}_{analysis_type}"
                df_key = f"{data_type}_{analysis_type}"
                self._write_mean_std(self.features_commentlevel, feature_key, df_key, self.comment_analysis_df)

        return self.features_commentlevel

    def _get_minmaxmost_words(self, df, data_key) -> dict:
        return {
            "max_word_score": df.at[df[data_key].idxmax(), data_key],
            "min_word_score": df.at[df[data_key].idxmin(), data_key],
            "most_word_score": df.at[df['Count'].idxmax(), data_key],
        }

    def _write_mean_std(self, dict, write_key, wordlist_key, df):
        data = get_mean_std(df[wordlist_key], len(df))
        dict[f"{write_key}_mean"] = data[0]
        dict[f"{write_key}_std"] = data[1]

    def _ratio(self, series, threshold):
        return sum(n > threshold for n in series) / sum(n < threshold for n in series)

    def _write_ratio(self, dict, write_key, series_key, threshold, df):
        dict[f"{write_key}_ratio"] = self._ratio(df[series_key], threshold)

    def _write_min(self, dict, df, write_key, wordlist_key):
        dict[f"{write_key}_min_word"] = df.at[df[wordlist_key].idxmin(), wordlist_key]

    def _write_max(self, dict, df, write_key, wordlist_key):
        dict[f"{write_key}_max_word"] = df.at[df[wordlist_key].idxmax(), wordlist_key]

    def _write_most(self, dict, df, write_key, wordlist_key):
        dict[f"{write_key}_most_word"] = df.at[df['Count'].idxmax(), wordlist_key]

    def _itervad(self):
        for data_type in [('v', 'Valence'), ('a', 'Arousal'), ('d', 'Dominance')]:
            yield data_type[0], data_type[1]

        