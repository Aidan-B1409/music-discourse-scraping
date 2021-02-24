import pandas as pd 

from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df
from os import getcwd

combinations = ['EmoVad_^_EmoLex', 'EmoVad_^_MPQA', 'BsmVad_^_EmoLex', 'BsmVad_^_MPQA']
affects = ['positive', 'negative']


def get_glob_headers() -> dict:
    dict = {}
    for dataset_combo in combinations:
        for datatype in ['v', 'a', 'd']:
            for uniq_status in ['uniq_', '']:
                for affect_type in affects:
                    for analysis_type in ['mean', 'std']:
                        dict[f"{dataset_combo}_glob_{datatype}_{affect_type}_{uniq_status}{analysis_type}"] = 0.0
                    for analysis_type in ['max_word', 'min_word', 'most_word']:
                        dict[f"{dataset_combo}_glob_{datatype}_{affect_type}_{analysis_type}"] = 0.0
                dict[f'{dataset_combo}_glob_{datatype}_{uniq_status}ratio'] = 0.0
        
    return dict

def get_commentlevel_headers() -> dict:
    dict = {}
    for dataset_combo in combinations:
        for affect_type in affects:
            for datatype in ['v', 'a', 'd']:
                for analysis_type in ['means', 'stds']:
                    for uniq_status in ['_uniq', '']:
                        for value in ['mean', 'std']:
                            dict[f'{dataset_combo}_{datatype}_{affect_type}{uniq_status}_{analysis_type}_{value}'] = 0.0
                for analysis_type in ['max_words', 'min_words', 'most_words']:
                    for value in ['mean', 'std']:
                        dict[f'{dataset_combo}_{datatype}_{affect_type}_{analysis_type}_{value}'] = 0.0
    return dict

def get_header():
    header = {}
    header.update(get_glob_headers())
    header.update(get_commentlevel_headers())
    return header
    

class MultiDataset_wlist:
 
    def __init__(self, wordlists: dict) -> None:
        self.comment_analysis_dfs = {key: pd.DataFrame() for key in combinations}
        for dataset in combinations:
            columns = []
            for affect_type in affects:
                for datatype in ['v', 'a', 'd']:
                    for analysis in ['means', 'stds']:
                        for uniq_status in ['_uniq', '']:
                            columns.append(f"{datatype}_{affect_type}{uniq_status}_{analysis}")
                    for analysis in ['max_words', 'min_words', 'most_words']:
                        columns.append(f"{datatype}_{affect_type}_{analysis}")
            self.comment_analysis_dfs[dataset] = pd.DataFrame(columns=columns)

        self.features_wordlevel = get_glob_headers()
        self.features_commentlevel = get_commentlevel_headers()

        self.emovad_df = wordlists['EmoVAD'].copy().set_index('Word')
        self.emolex_df = wordlists['EmoLex'].copy().set_index('Word')
        self.mpqa_df = wordlists['MPQA'].copy().set_index('Word')
        self.bsmvad_df = wordlists['ANEW_Extended'].copy().set_index('Word')


        columns = {'V.Mean.Sum': 'Valence',
                    'A.Mean.Sum': 'Arousal',
                    'D.Mean.Sum': 'Dominance'}
        self.bsmvad_df = self.bsmvad_df.rename(columns = columns)

        # WARNING - very tightly coupled to global 'combinations'
        # Check there before changing 
        # Side effects may include - line 77 totally breaking
        self.combos_tuple = [(self.emovad_df, self.emolex_df), (self.emovad_df, self.mpqa_df),
        (self.bsmvad_df, self.emolex_df), (self.bsmvad_df, self.mpqa_df)]
   
        
    def wordlevel_analysis(self, song_df, glob_df) -> dict:
        if(len(song_df) > 0):
            for index, datasets in enumerate(self.combos_tuple): 
                root = combinations[index] + "_glob"

                positive_words = self._get_affect_subset('positive', datasets[1]) if (datasets[1] is self.emolex_df) else self._get_mpqa_sentiment_subset('positive', datasets[1])
                negative_words = self._get_affect_subset('negative', datasets[1]) if (datasets[1] is self.emolex_df) else self._get_mpqa_sentiment_subset('negative', datasets[1])
                # positive_words.set_index('Word', inplace=True)
                sentiment_dfs = {'positive': positive_words.join(datasets[0], how='inner'), 'negative': negative_words.join(datasets[0], how='inner')}

                for data_type, data_key in self._itervad():
                    sentiment_counts = {"positive": 0, "negative":0}
                    uniq_sentiment_counts = {"positive": 0, "negative":0}

                    for affect_type in affects:
                        vad_wordlist = sentiment_dfs[affect_type]

                        matched_words = unsquished_intersection(song_df, vad_wordlist)
                        matched_uniq_words = glob_intersection(glob_df, vad_wordlist)

                        self._write_mean_std(self.features_wordlevel, f"{root}_{data_type}_{affect_type}", data_key, matched_words)
                                            
                        self._write_mean_std(self.features_wordlevel, f"{root}_{data_type}_{affect_type}_uniq", data_key, matched_uniq_words)
                        
                        if(len(matched_uniq_words) > 0):
                            self._write_min(self.features_wordlevel, matched_uniq_words, f"{root}_{data_type}_{affect_type}", data_key)
                            self._write_max(self.features_wordlevel, matched_uniq_words, f"{root}_{data_type}_{affect_type}", data_key)
                            self._write_most(self.features_wordlevel, matched_uniq_words, f"{root}_{data_type}_{affect_type}", data_key)

                        sentiment_counts[affect_type] = len(matched_words)
                        uniq_sentiment_counts[affect_type] = len(matched_uniq_words)
               
                    if(uniq_sentiment_counts['negative'] > 0):
                        self.features_wordlevel[f'{root}_{data_type}_ratio'] = sentiment_counts['positive'] / sentiment_counts['negative']
                        self.features_wordlevel[f'{root}_{data_type}_uniq_ratio'] = uniq_sentiment_counts['positive'] / uniq_sentiment_counts['negative']

        return self.features_wordlevel

    def process_comment(self, index, comment):

        comment_list = clean_comment(comment)

        unique_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])
        words_df.set_index('Word', inplace=True)

        for dfidx, analysis_df in enumerate(self.comment_analysis_dfs.values()):
            root = combinations[dfidx]
            datasets = self.combos_tuple[dfidx]
            
            for affect_type in affects:
                affect_wordlist = self._get_affect_subset(affect_type, datasets[1]) if (datasets[1] is self.emolex_df) else self._get_mpqa_sentiment_subset(affect_type, datasets[1])
                #vad_affect_wordlist = pd.merge(affect_wordlist, datasets[0])
                vad_affect_wordlist = affect_wordlist.join(datasets[0], how='inner')

                found_uniq_word_df = glob_intersection(unique_words_df, vad_affect_wordlist)
                #found_words_df = pd.merge(words_df, vad_affect_wordlist, on='Word')
                found_words_df = words_df.join(vad_affect_wordlist, how='inner')

                for data_type, data_key in self._itervad():
                    data = get_mean_std(found_words_df[data_key], len(found_words_df))
                    analysis_df.at[index, f"{data_type}_{affect_type}_means"] = data[0]
                    analysis_df.at[index, f"{data_type}_{affect_type}_stds"] = data[1]

                    uniq_data = get_mean_std(found_uniq_word_df[data_key], len(found_uniq_word_df))
                    analysis_df.at[index, f"{data_type}_{affect_type}_uniq_means"] = uniq_data[0]
                    analysis_df.at[index, f"{data_type}_{affect_type}_uniq_stds"] = uniq_data[1]

                    if(len(found_uniq_word_df) > 0):
                        minmaxmost = self._get_minmaxmost_words(found_uniq_word_df, data_key)
                        analysis_df.at[index, f"{data_type}_{affect_type}_max_words"] = minmaxmost['max_word_score']
                        analysis_df.at[index, f"{data_type}_{affect_type}_min_words"] = minmaxmost['min_word_score']
                        analysis_df.at[index, f"{data_type}_{affect_type}_most_words"] = minmaxmost['most_word_score']


    def analyze_comments(self) -> dict:
        for analysis_df_key, analysis_df in self.comment_analysis_dfs.items():
            for col in analysis_df:
                self._write_mean_std(self.features_commentlevel, f'{analysis_df_key}_{col}', col, analysis_df)
        return self.features_commentlevel

    def _get_affect_subset(self, affect_key, emolex_df) -> pd.DataFrame:
        emolex_affect_subset = emolex_df[(emolex_df['Emotion'] == affect_key)].copy()
        indexes = emolex_affect_subset[emolex_affect_subset['Association'] == 0].index
        emolex_affect_subset.drop(indexes, inplace=True)
        return emolex_affect_subset

    def _get_mpqa_sentiment_subset(self, affect_key, df) -> pd.DataFrame:
        excluded_indexes = df[df['Sentiment'] != affect_key].index
        return df.drop(excluded_indexes).copy()

    def _itervad(self):
        for data_type in [('v', 'Valence'), ('a', 'Arousal'), ('d', 'Dominance')]:
            yield data_type[0], data_type[1]

    def _get_minmaxmost_words(self, df, data_key) -> dict:
        return {
            "max_word_score": df.at[df[data_key].idxmax(), data_key],
            "min_word_score": df.at[df[data_key].idxmin(), data_key],
            "most_word_score": df.at[df['Count'].idxmax(), data_key],
        }

    def _write_mean_std(self, dict, write_key, wordlist_key, df):
        series = df[wordlist_key].dropna()
        data = get_mean_std(series, len(series))
        dict[f"{write_key}_mean"] = data[0]
        dict[f"{write_key}_std"] = data[1]

    def _write_min(self, dict, df, write_key, wordlist_key):
        dict[f"{write_key}_min_word"] = df.at[df[wordlist_key].idxmin(), wordlist_key]

    def _write_max(self, dict, df, write_key, wordlist_key):
        dict[f"{write_key}_max_word"] = df.at[df[wordlist_key].idxmax(), wordlist_key]

    def _write_most(self, dict, df, write_key, wordlist_key):
        dict[f"{write_key}_most_word"] = df.at[df['Count'].idxmax(), wordlist_key]
