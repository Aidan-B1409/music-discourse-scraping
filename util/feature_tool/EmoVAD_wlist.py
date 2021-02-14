import pandas as pd 
import numpy as np

from string_cleaner import clean_comment
from string_cleaner import make_word_df
from glob_maker import make_unsquished_glob

from statistics import stdev


class EmoVAD_wlist:


    def __init__(self, wlist_path: str) -> None:
        columns = ['valence_means', 'valence_stds', 'valence_uniq_means', 'valence_uniq_stds',
                'arousal_means', 'arousal_stds', 'arousal_uniq_means', 'arousal_uniq_stds',
                'dominance_means', 'dominance_stds', 'dominance_uniq_means', 'dominance_uniq_stds',
                'max_word_v', "max_word_a", "max_word_d", "min_word_v", "min_word_a", "min_word_d",
                "most_word_v", "most_word_a", "most_word_d", "most_word_count"]
        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.emovad_df = pd.read_csv(wlist_path, names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t')

        self.features_wordlevel = {
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
        }

        self.features_commentlevel = {
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
        }

    def wordlevel_analysis(self, song_df, glob_df) -> dict:

        if(len(song_df) > 0):
            allword_semantic_word_df = self._song_df_intersection(song_df, self.emovad_df)
            uniq_semantic_word_df = self._make_glob_intersection(glob_df, self.emovad_df)

            self._write_mean_std(self.features_wordlevel, "EmoVAD_glob_v_mean", "EmoVAD_glob_v_stdev", self._get_mean_std(allword_semantic_word_df['Valence'], len(allword_semantic_word_df['Word'])))
            self._write_mean_std(self.features_wordlevel, "EmoVAD_glob_a_mean", "EmoVAD_glob_a_stdev", self._get_mean_std(allword_semantic_word_df['Arousal'], len(allword_semantic_word_df['Word'])))
            self._write_mean_std(self.features_wordlevel, "EmoVAD_glob_d_mean", "EmoVAD_glob_d_stdev", self._get_mean_std(allword_semantic_word_df['Dominance'], len(allword_semantic_word_df['Word'])))

            self.features_wordlevel['EmoVAD_glob_v_sqmean'] = np.square(allword_semantic_word_df['Valence']).mean()
            self.features_wordlevel['EmoVAD_glob_a_sqmean'] = np.square(allword_semantic_word_df['Arousal']).mean()
            self.features_wordlevel['EmoVAD_glob_d_sqmean'] = np.square(allword_semantic_word_df['Dominance']).mean()
            self.features_wordlevel['EmoVAD_glob_v_sqmean_uniq'] = np.square(allword_semantic_word_df['Valence']).mean()
            self.features_wordlevel['EmoVAD_glob_a_sqmean_uniq'] = np.square(allword_semantic_word_df['Arousal']).mean()
            self.features_wordlevel['EmoVAD_glob_d_sqmean_uniq'] = np.square(allword_semantic_word_df['Dominance']).mean()
            #VAD means on unique words only
            self._write_mean_std(self.features_wordlevel, "EmoVAD_glob_v_mean_uniq", "EmoVAD_glob_v_stdev_uniq", self._get_mean_std(uniq_semantic_word_df['Valence'], len(uniq_semantic_word_df['Word'])))
            self._write_mean_std(self.features_wordlevel, "EmoVAD_glob_a_mean_uniq", "EmoVAD_glob_a_stdev_uniq", self._get_mean_std(uniq_semantic_word_df['Arousal'], len(uniq_semantic_word_df['Word'])))
            self._write_mean_std(self.features_wordlevel, "EmoVAD_glob_d_mean_uniq", "EmoVAD_glob_d_stdev_uniq", self._get_mean_std(uniq_semantic_word_df['Dominance'], len(uniq_semantic_word_df['Word'])))

            if(len(uniq_semantic_word_df) > 0):
                self._write_minmaxmost_features(uniq_semantic_word_df)
        return self.features_wordlevel

    def process_comment(self, index, comment):
        # clean the string
        comment_list = clean_comment(comment)

        unique_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])

        semantic_uniq_word_df = self._make_glob_intersection(unique_words_df, self.emovad_df)
        semantic_words_df = pd.merge(words_df, self.emovad_df, on='Word')

        # If there are any words which match our VAD wordlist 
        if len(unique_words_df) > 0:

            v_data = self._get_mean_std(semantic_words_df['Valence'], len(semantic_words_df['Word']))
            self.comment_analysis_df.at[index,'valence_means'] = v_data[0]
            self.comment_analysis_df.at[index,'valence_stds'] = v_data[1]

            v_data_uniq = self._get_mean_std(semantic_uniq_word_df['Valence'], len(semantic_uniq_word_df['Word']))
            self.comment_analysis_df.at[index,'valence_uniq_means'] = v_data_uniq[0]
            self.comment_analysis_df.at[index,'valence_uniq_stds'] = v_data_uniq[1]

            a_data = self._get_mean_std(semantic_words_df['Arousal'], len(semantic_words_df['Word']))
            self.comment_analysis_df.at[index,'arousal_means'] = a_data[0]
            self.comment_analysis_df.at[index,'arousal_stds'] = a_data[1]

            a_data_uniq = self._get_mean_std(semantic_uniq_word_df['Arousal'], len(semantic_uniq_word_df['Word']))
            self.comment_analysis_df.at[index,'arousal_uniq_means'] = a_data_uniq[0]
            self.comment_analysis_df.at[index,'arousal_uniq_stds'] = a_data_uniq[1]

            d_data = self._get_mean_std(semantic_words_df['Dominance'], len(semantic_words_df['Word']))
            self.comment_analysis_df.at[index,'dominance_means'] = d_data[0]
            self.comment_analysis_df.at[index,'dominance_stds'] = d_data[1]

            d_data_uniq = self._get_mean_std(semantic_uniq_word_df['Dominance'], len(semantic_uniq_word_df['Word']))
            self.comment_analysis_df.at[index,'dominance_uniq_means'] = d_data_uniq[0]
            self.comment_analysis_df.at[index,'dominance_uniq_stds'] = d_data_uniq[1]

            if(len(semantic_uniq_word_df) > 0):
                words = self._get_minmaxmost_words(semantic_uniq_word_df)
                self.comment_analysis_df.at[index, 'max_word_v'] = semantic_uniq_word_df.at[words['highest_valence_word'], 'Valence']
                self.comment_analysis_df.at[index, 'max_word_a'] = semantic_uniq_word_df.at[words['highest_arousal_word'], 'Arousal']
                self.comment_analysis_df.at[index, 'max_word_d'] = semantic_uniq_word_df.at[words['highest_dominance_word'], 'Dominance']

                self.comment_analysis_df.at[index, 'min_word_v'] = semantic_uniq_word_df.at[words['lowest_valence_word'], 'Valence']
                self.comment_analysis_df.at[index, 'min_word_a'] = semantic_uniq_word_df.at[words['lowest_arousal_word'], 'Arousal']
                self.comment_analysis_df.at[index, 'min_word_d'] = semantic_uniq_word_df.at[words['lowest_dominance_word'], 'Dominance']

                self.comment_analysis_df.at[index, 'most_word_v'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'Valence']
                self.comment_analysis_df.at[index, 'most_word_a'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'Arousal']
                self.comment_analysis_df.at[index, 'most_word_d'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'Dominance']
                self.comment_analysis_df.at[index, 'most_word_count'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'Count']

    def analyze_comments(self) -> dict:
        self._write_mean_std(self.features_commentlevel, "EmoVAD_v_mean_mean", "EmoVAD_v_mean_stdev", self._get_mean_std(self.comment_analysis_df['valence_means'], len(self.comment_analysis_df['valence_means'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_v_std_mean", "EmoVAD_v_stdev_stdev", self._get_mean_std(self.comment_analysis_df['valence_stds'], len(self.comment_analysis_df['valence_stds'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_v_uniq_mean_mean", "EmoVAD_v_uniq_mean_stdev", self._get_mean_std(self.comment_analysis_df['valence_uniq_means'], len(self.comment_analysis_df['valence_uniq_means'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_v_uniq_std_mean", "EmoVAD_v_uniq_stdev_stdev", self._get_mean_std(self.comment_analysis_df['valence_uniq_stds'], len(self.comment_analysis_df['valence_uniq_stds'])))
        
        self._write_mean_std(self.features_commentlevel, "EmoVAD_a_mean_mean", "EmoVAD_a_mean_stdev", self._get_mean_std(self.comment_analysis_df['arousal_means'], len(self.comment_analysis_df['arousal_means'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_a_std_mean", "EmoVAD_a_stdev_stdev", self._get_mean_std(self.comment_analysis_df['arousal_stds'], len(self.comment_analysis_df['arousal_stds'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_a_uniq_mean_mean", "EmoVAD_a_uniq_mean_stdev", self._get_mean_std(self.comment_analysis_df['arousal_uniq_means'], len(self.comment_analysis_df['arousal_uniq_means'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_a_uniq_std_mean", "EmoVAD_a_uniq_stdev_stdev", self._get_mean_std(self.comment_analysis_df['arousal_uniq_stds'], len(self.comment_analysis_df['arousal_uniq_stds'])))
        
        self._write_mean_std(self.features_commentlevel, "EmoVAD_d_mean_mean", "EmoVAD_d_mean_stdev", self._get_mean_std(self.comment_analysis_df['dominance_means'], len(self.comment_analysis_df['dominance_means'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_d_std_mean", "EmoVAD_d_stdev_stdev", self._get_mean_std(self.comment_analysis_df['dominance_stds'], len(self.comment_analysis_df['dominance_stds'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_d_uniq_mean_mean", "EmoVAD_d_uniq_mean_stdev", self._get_mean_std(self.comment_analysis_df['dominance_uniq_means'], len(self.comment_analysis_df['dominance_uniq_means'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_d_uniq_std_mean", "EmoVAD_d_uniq_stdev_stdev", self._get_mean_std(self.comment_analysis_df['dominance_uniq_stds'], len(self.comment_analysis_df['dominance_uniq_stds'])))

        self._write_mean_std(self.features_commentlevel, "EmoVAD_max_word_v_mean", "EmoVAD_max_word_v_std", self._get_mean_std(self.comment_analysis_df['max_word_v'], len(self.comment_analysis_df['max_word_v'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_max_word_a_mean", "EmoVAD_max_word_a_std", self._get_mean_std(self.comment_analysis_df['max_word_a'], len(self.comment_analysis_df['max_word_a'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_max_word_d_mean", "EmoVAD_max_word_d_std", self._get_mean_std(self.comment_analysis_df['max_word_d'], len(self.comment_analysis_df['max_word_d'])))

        self._write_mean_std(self.features_commentlevel, "EmoVAD_max_word_v_sqmean", "EmoVAD_max_word_v_sqstd", self._get_mean_std(np.square(self.comment_analysis_df['max_word_v']), len(self.comment_analysis_df['max_word_v'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_max_word_a_sqmean", "EmoVAD_max_word_a_sqstd", self._get_mean_std(np.square(self.comment_analysis_df['max_word_a']), len(self.comment_analysis_df['max_word_a'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_max_word_d_sqmean", "EmoVAD_max_word_d_sqstd", self._get_mean_std(np.square(self.comment_analysis_df['max_word_d']), len(self.comment_analysis_df['max_word_d'])))

        self._write_mean_std(self.features_commentlevel, "EmoVAD_min_word_v_mean", "EmoVAD_min_word_v_std", self._get_mean_std(self.comment_analysis_df['min_word_v'], len(self.comment_analysis_df['min_word_v'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_min_word_a_mean", "EmoVAD_min_word_a_std", self._get_mean_std(self.comment_analysis_df['min_word_a'], len(self.comment_analysis_df['min_word_a'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_min_word_d_mean", "EmoVAD_min_word_d_std", self._get_mean_std(self.comment_analysis_df['min_word_d'], len(self.comment_analysis_df['min_word_d'])))

        self._write_mean_std(self.features_commentlevel, "EmoVAD_most_word_v_mean", "EmoVAD_most_word_v_std", self._get_mean_std(self.comment_analysis_df['most_word_v'], len(self.comment_analysis_df['most_word_v'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_most_word_a_mean", "EmoVAD_most_word_a_std", self._get_mean_std(self.comment_analysis_df['most_word_a'], len(self.comment_analysis_df['most_word_a'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_most_word_d_mean", "EmoVAD_most_word_d_std", self._get_mean_std(self.comment_analysis_df['most_word_d'], len(self.comment_analysis_df['most_word_d'])))
        self._write_mean_std(self.features_commentlevel, "EmoVAD_most_word_count_mean", "EmoVAD_most_word_count_std", self._get_mean_std(self.comment_analysis_df['most_word_count'], len(self.comment_analysis_df['most_word_count'])))
        return self.features_commentlevel

    def _write_mean_std(self, features_dict, mean_key, std_key, data: tuple) -> None:
        features_dict[mean_key] = data[0]
        features_dict[std_key] = data [1]


    def _write_minmaxmost_features(self, semantic_word_df):
        words = self.get_minmaxmost_words(semantic_word_df)
        
        self.features_wordlevel["EmoVAD_glob_max_word_v"] = semantic_word_df.at[words['highest_valence_word'], "Valence"]
        self.features_wordlevel["EmoVAD_glob_max_word_a"] = semantic_word_df.at[words['highest_arousal_word'], "Arousal"]
        self.features_wordlevel["EmoVAD_glob_max_word_d"] = semantic_word_df.at[words['highest_dominance_word'], "Dominance"]

        self.features_wordlevel["EmoVAD_glob_min_word_v"] = semantic_word_df.at[words['lowest_valence_word'], "Valence"]
        self.features_wordlevel["EmoVAD_glob_min_word_a"] = semantic_word_df.at[words['lowest_arousal_word'], "Arousal"]
        self.features_wordlevel["EmoVAD_glob_min_word_d"] = semantic_word_df.at[words['lowest_dominance_word'], "Dominance"]
        
        self.features_wordlevel["EmoVAD_glob_most_word_v"] = semantic_word_df.at[words['most_occuring_word'], "Valence"]
        self.features_wordlevel["EmoVAD_glob_most_word_a"] = semantic_word_df.at[words['most_occuring_word'], "Arousal"]
        self.features_wordlevel["EmoVAD_glob_most_word_d"] = semantic_word_df.at[words['most_occuring_word'], "Dominance"]
        self.features_wordlevel["EmoVAD_glob_most_word_count"] = semantic_word_df.at[words['most_occuring_word'], "Count"]

    # Find the mean and standard deviation of a given series from a dataframe, with a length unbound from the dataframe itself
    # returns (mean, std) tuple
    def _get_mean_std(self, series: pd.Series, length) -> tuple:
        if length > 0:
            mean = series.sum() / length
            std = self._std_check(series, mean)
            return (mean, std)
        return (0, 0)

    def _std_check(self, series, mean) -> float:
        if(len(series) >= 2):
            return stdev(series, xbar = mean)
        else:
            return 0

    def _get_minmaxmost_words(self, semantic_word_df) -> dict:
        return {
            "highest_valence_word": semantic_word_df['Valence'].idxmax(),
            "highest_arousal_word": semantic_word_df['Arousal'].idxmax(),
            "highest_dominance_word": semantic_word_df['Dominance'].idxmax(),
            "lowest_valence_word": semantic_word_df['Valence'].idxmin(), 
            "lowest_arousal_word": semantic_word_df['Arousal'].idxmin(), 
            "lowest_dominance_word": semantic_word_df['Dominance'].idxmin(),
            "most_occuring_word": semantic_word_df['Count'].idxmax()
        }

    def get_minmaxmost_words(self, semantic_word_df) -> dict:
        return {
            "highest_valence_word": semantic_word_df['Valence'].idxmax(),
            "highest_arousal_word": semantic_word_df['Arousal'].idxmax(),
            "highest_dominance_word": semantic_word_df['Dominance'].idxmax(),
            "lowest_valence_word": semantic_word_df['Valence'].idxmin(), 
            "lowest_arousal_word": semantic_word_df['Arousal'].idxmin(), 
            "lowest_dominance_word": semantic_word_df['Dominance'].idxmin(),
            "most_occuring_word": semantic_word_df['Count'].idxmax()
        }

    def _make_glob_intersection(self, glob_df, wordlist_df) -> pd.DataFrame:
        semantic_word_df = pd.merge(glob_df, wordlist_df, on='Word')

        semantic_word_df['V_Total'] = semantic_word_df['Count'] * semantic_word_df['Valence']
        semantic_word_df['A_Total'] = semantic_word_df['Count'] * semantic_word_df['Arousal']
        semantic_word_df['D_Total'] = semantic_word_df['Count'] * semantic_word_df['Dominance']
        return semantic_word_df


    def _song_df_intersection(self, song_df, wordlist_df) -> pd.DataFrame:
        semantic_wordbag_df = pd.merge(make_unsquished_glob(song_df), wordlist_df, on='Word')
        return semantic_wordbag_df

        