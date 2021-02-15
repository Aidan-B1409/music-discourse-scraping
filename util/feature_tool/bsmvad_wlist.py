import pandas as pd 
import numpy as np

from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

class BSMVAD_wlist:

    def __init__(self, wlist_path: str) -> None:
        columns = ['valence_means', 'valence_stds', 'valence_uniq_means', 'valence_uniq_stds',
                'arousal_means', 'arousal_stds', 'arousal_uniq_means', 'arousal_uniq_stds',
                'dominance_means', 'dominance_stds', 'dominance_uniq_means', 'dominance_uniq_stds',
                'max_word_v', "max_word_a", "max_word_d", "min_word_v", "min_word_a", "min_word_d",
                "most_word_v", "most_word_a", "most_word_d", "most_word_count"]
        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.bsmvad_df = pd.read_csv(wlist_path, encoding='utf-8', engine='python')
        # drop unneeded columns
        self.bsmvad_df.drop(self.bsmvad_df.iloc[:, 10:64].columns, axis = 1, inplace = True) 
        self.bsmvad_df.drop(['V.Rat.Sum', 'A.Rat.Sum','D.Rat.Sum'], axis = 1, inplace = True) 
        # drop blank rows, if any
        self.bsmvad_df = self.bsmvad_df[self.bsmvad_df['Word'].notnull()]

        self.features_wordlevel = {
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
        }

        self.features_commentlevel = {
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


    def wordlevel_analysis(self, song_df, glob_df) -> dict:
        if(len(song_df) > 0):
            allword_semantic_word_df = unsquished_intersection(song_df, self.bsmvad_df)
            uniq_semantic_word_df = glob_intersection(glob_df, self.bsmvad_df)

            if(len(allword_semantic_word_df) > 0):
                write_mean_std(self.features_wordlevel, "bsmVAD_glob_v_mean", "bsmVAD_glob_v_stdev", get_mean_std(allword_semantic_word_df['V.Mean.Sum'], len(allword_semantic_word_df['Word'])))
                write_mean_std(self.features_wordlevel, "bsmVAD_glob_a_mean", "bsmVAD_glob_a_stdev", get_mean_std(allword_semantic_word_df['A.Mean.Sum'], len(allword_semantic_word_df['Word'])))
                write_mean_std(self.features_wordlevel, "bsmVAD_glob_d_mean", "bsmVAD_glob_d_stdev", get_mean_std(allword_semantic_word_df['D.Mean.Sum'], len(allword_semantic_word_df['Word'])))

                self.features_wordlevel['bsmVAD_glob_v_sqmean'] = np.square(allword_semantic_word_df['V.Mean.Sum']).mean()
                self.features_wordlevel['bsmVAD_glob_a_sqmean'] = np.square(allword_semantic_word_df['A.Mean.Sum']).mean()
                self.features_wordlevel['bsmVAD_glob_d_sqmean'] = np.square(allword_semantic_word_df['D.Mean.Sum']).mean()

                # TODO - map to 0..1 range? 
                self.features_wordlevel['bsmVAD_glob_v_ratio'] = ratio(allword_semantic_word_df['V.Mean.Sum'], self.features_wordlevel['bsmVAD_glob_v_mean'])
                self.features_wordlevel['bsmVAD_glob_a_ratio'] = ratio(allword_semantic_word_df['A.Mean.Sum'], self.features_wordlevel['bsmVAD_glob_a_mean'])
                self.features_wordlevel['bsmVAD_glob_d_ratio'] = ratio(allword_semantic_word_df['D.Mean.Sum'], self.features_wordlevel['bsmVAD_glob_d_mean'])

            if(len(uniq_semantic_word_df)):
                write_mean_std(self.features_wordlevel, "bsmVAD_glob_v_mean_uniq", "bsmVAD_glob_v_stdev_uniq", get_mean_std(uniq_semantic_word_df['V.Mean.Sum'], len(uniq_semantic_word_df['Word'])))
                write_mean_std(self.features_wordlevel, "bsmVAD_glob_a_mean_uniq", "bsmVAD_glob_a_stdev_uniq", get_mean_std(uniq_semantic_word_df['A.Mean.Sum'], len(uniq_semantic_word_df['Word'])))
                write_mean_std(self.features_wordlevel, "bsmVAD_glob_d_mean_uniq", "bsmVAD_glob_d_stdev_uniq", get_mean_std(uniq_semantic_word_df['D.Mean.Sum'], len(uniq_semantic_word_df['Word'])))

                self.features_wordlevel['bsmVAD_glob_v_sqmean_uniq'] = np.square(uniq_semantic_word_df['V.Mean.Sum']).mean()
                self.features_wordlevel['bsmVAD_glob_a_sqmean_uniq'] = np.square(uniq_semantic_word_df['A.Mean.Sum']).mean()
                self.features_wordlevel['bsmVAD_glob_d_sqmean_uniq'] = np.square(uniq_semantic_word_df['D.Mean.Sum']).mean()
                self._write_minmaxmost_features(uniq_semantic_word_df)

                # TODO - map to 0..1 range? 
                self.features_wordlevel['bsmVAD_glob_v_uniq_ratio'] = ratio(uniq_semantic_word_df['V.Mean.Sum'], self.features_wordlevel['bsmVAD_glob_v_mean_uniq'])
                self.features_wordlevel['bsmVAD_glob_a_uniq_ratio'] = ratio(uniq_semantic_word_df['A.Mean.Sum'], self.features_wordlevel['bsmVAD_glob_a_mean_uniq'])
                self.features_wordlevel['bsmVAD_glob_d_uniq_ratio'] = ratio(uniq_semantic_word_df['D.Mean.Sum'], self.features_wordlevel['bsmVAD_glob_d_mean_uniq'])

        return self.features_wordlevel

    def process_comment(self, index, comment):
        # clean the string
        comment_list = clean_comment(comment)

        unique_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])

        semantic_uniq_word_df = glob_intersection(unique_words_df, self.bsmvad_df)
        semantic_words_df = pd.merge(words_df, self.bsmvad_df, on='Word')

        # If there are any words which match our VAD wordlist 
        if len(unique_words_df) > 0:

            v_data = get_mean_std(semantic_words_df['V.Mean.Sum'], len(semantic_words_df['Word']))
            self.comment_analysis_df.at[index,'valence_means'] = v_data[0]
            self.comment_analysis_df.at[index,'valence_stds'] = v_data[1]

            v_data_uniq = get_mean_std(semantic_uniq_word_df['V.Mean.Sum'], len(semantic_uniq_word_df['Word']))
            self.comment_analysis_df.at[index,'valence_uniq_means'] = v_data_uniq[0]
            self.comment_analysis_df.at[index,'valence_uniq_stds'] = v_data_uniq[1]

            a_data = get_mean_std(semantic_words_df['A.Mean.Sum'], len(semantic_words_df['Word']))
            self.comment_analysis_df.at[index,'arousal_means'] = a_data[0]
            self.comment_analysis_df.at[index,'arousal_stds'] = a_data[1]

            a_data_uniq = get_mean_std(semantic_uniq_word_df['A.Mean.Sum'], len(semantic_uniq_word_df['Word']))
            self.comment_analysis_df.at[index,'arousal_uniq_means'] = a_data_uniq[0]
            self.comment_analysis_df.at[index,'arousal_uniq_stds'] = a_data_uniq[1]

            d_data = get_mean_std(semantic_words_df['D.Mean.Sum'], len(semantic_words_df['Word']))
            self.comment_analysis_df.at[index,'dominance_means'] = d_data[0]
            self.comment_analysis_df.at[index,'dominance_stds'] = d_data[1]

            d_data_uniq = get_mean_std(semantic_uniq_word_df['D.Mean.Sum'], len(semantic_uniq_word_df['Word']))
            self.comment_analysis_df.at[index,'dominance_uniq_means'] = d_data_uniq[0]
            self.comment_analysis_df.at[index,'dominance_uniq_stds'] = d_data_uniq[1]

        if(len(semantic_uniq_word_df) > 0):
            words = self._get_minmaxmost_words(semantic_uniq_word_df)
            self.comment_analysis_df.at[index, 'max_word_v'] = semantic_uniq_word_df.at[words['highest_valence_word'], 'V.Mean.Sum']
            self.comment_analysis_df.at[index, 'max_word_a'] = semantic_uniq_word_df.at[words['highest_arousal_word'], 'A.Mean.Sum']
            self.comment_analysis_df.at[index, 'max_word_d'] = semantic_uniq_word_df.at[words['highest_dominance_word'], 'D.Mean.Sum']

            self.comment_analysis_df.at[index, 'min_word_v'] = semantic_uniq_word_df.at[words['lowest_valence_word'], 'V.Mean.Sum']
            self.comment_analysis_df.at[index, 'min_word_a'] = semantic_uniq_word_df.at[words['lowest_arousal_word'], 'A.Mean.Sum']
            self.comment_analysis_df.at[index, 'min_word_d'] = semantic_uniq_word_df.at[words['lowest_dominance_word'], 'D.Mean.Sum']

            self.comment_analysis_df.at[index, 'most_word_v'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'V.Mean.Sum']
            self.comment_analysis_df.at[index, 'most_word_a'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'A.Mean.Sum']
            self.comment_analysis_df.at[index, 'most_word_d'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'D.Mean.Sum']
            self.comment_analysis_df.at[index, 'most_word_count'] = semantic_uniq_word_df.at[words['most_occuring_word'], 'Count']

    def analyze_comments(self) -> dict:
        write_mean_std(self.features_commentlevel, "bsmVAD_v_mean_mean", "bsmVAD_v_mean_stdev", get_mean_std(self.comment_analysis_df['valence_means'], len(self.comment_analysis_df['valence_means'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_v_std_mean", "bsmVAD_v_stdev_stdev", get_mean_std(self.comment_analysis_df['valence_stds'], len(self.comment_analysis_df['valence_stds'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_v_uniq_mean_mean", "bsmVAD_v_uniq_mean_stdev", get_mean_std(self.comment_analysis_df['valence_uniq_means'], len(self.comment_analysis_df['valence_uniq_means'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_v_uniq_std_mean", "bsmVAD_v_uniq_stdev_stdev", get_mean_std(self.comment_analysis_df['valence_uniq_stds'], len(self.comment_analysis_df['valence_uniq_stds'])))
        
        write_mean_std(self.features_commentlevel, "bsmVAD_a_mean_mean", "bsmVAD_a_mean_stdev", get_mean_std(self.comment_analysis_df['arousal_means'], len(self.comment_analysis_df['arousal_means'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_a_std_mean", "bsmVAD_a_stdev_stdev", get_mean_std(self.comment_analysis_df['arousal_stds'], len(self.comment_analysis_df['arousal_stds'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_a_uniq_mean_mean", "bsmVAD_a_uniq_mean_stdev", get_mean_std(self.comment_analysis_df['arousal_uniq_means'], len(self.comment_analysis_df['arousal_uniq_means'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_a_uniq_std_mean", "bsmVAD_a_uniq_stdev_stdev", get_mean_std(self.comment_analysis_df['arousal_uniq_stds'], len(self.comment_analysis_df['arousal_uniq_stds'])))
        
        write_mean_std(self.features_commentlevel, "bsmVAD_d_mean_mean", "bsmVAD_d_mean_stdev", get_mean_std(self.comment_analysis_df['dominance_means'], len(self.comment_analysis_df['dominance_means'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_d_std_mean", "bsmVAD_d_stdev_stdev", get_mean_std(self.comment_analysis_df['dominance_stds'], len(self.comment_analysis_df['dominance_stds'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_d_uniq_mean_mean", "bsmVAD_d_uniq_mean_stdev", get_mean_std(self.comment_analysis_df['dominance_uniq_means'], len(self.comment_analysis_df['dominance_uniq_means'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_d_uniq_std_mean", "bsmVAD_d_uniq_stdev_stdev", get_mean_std(self.comment_analysis_df['dominance_uniq_stds'], len(self.comment_analysis_df['dominance_uniq_stds'])))

        write_mean_std(self.features_commentlevel, "bsmVAD_max_word_v_mean", "bsmVAD_max_word_v_std", get_mean_std(self.comment_analysis_df['max_word_v'], len(self.comment_analysis_df['max_word_v'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_max_word_a_mean", "bsmVAD_max_word_a_std", get_mean_std(self.comment_analysis_df['max_word_a'], len(self.comment_analysis_df['max_word_a'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_max_word_d_mean", "bsmVAD_max_word_d_std", get_mean_std(self.comment_analysis_df['max_word_d'], len(self.comment_analysis_df['max_word_d'])))

        write_mean_std(self.features_commentlevel, "bsmVAD_max_word_v_sqmean", "bsmVAD_max_word_v_sqstd", get_mean_std(np.square(self.comment_analysis_df['max_word_v']), len(self.comment_analysis_df['max_word_v'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_max_word_a_sqmean", "bsmVAD_max_word_a_sqstd", get_mean_std(np.square(self.comment_analysis_df['max_word_a']), len(self.comment_analysis_df['max_word_a'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_max_word_d_sqmean", "bsmVAD_max_word_d_sqstd", get_mean_std(np.square(self.comment_analysis_df['max_word_d']), len(self.comment_analysis_df['max_word_d'])))

        write_mean_std(self.features_commentlevel, "bsmVAD_min_word_v_mean", "bsmVAD_min_word_v_std", get_mean_std(self.comment_analysis_df['min_word_v'], len(self.comment_analysis_df['min_word_v'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_min_word_a_mean", "bsmVAD_min_word_a_std", get_mean_std(self.comment_analysis_df['min_word_a'], len(self.comment_analysis_df['min_word_a'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_min_word_d_mean", "bsmVAD_min_word_d_std", get_mean_std(self.comment_analysis_df['min_word_d'], len(self.comment_analysis_df['min_word_d'])))

        write_mean_std(self.features_commentlevel, "bsmVAD_most_word_v_mean", "bsmVAD_most_word_v_std", get_mean_std(self.comment_analysis_df['most_word_v'], len(self.comment_analysis_df['most_word_v'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_most_word_a_mean", "bsmVAD_most_word_a_std", get_mean_std(self.comment_analysis_df['most_word_a'], len(self.comment_analysis_df['most_word_a'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_most_word_d_mean", "bsmVAD_most_word_d_std", get_mean_std(self.comment_analysis_df['most_word_d'], len(self.comment_analysis_df['most_word_d'])))
        write_mean_std(self.features_commentlevel, "bsmVAD_most_word_count_mean", "bsmVAD_most_word_count_std", get_mean_std(self.comment_analysis_df['most_word_count'], len(self.comment_analysis_df['most_word_count'])))

        return self.features_commentlevel

    def _write_minmaxmost_features(self, semantic_word_df):
        words = self._get_minmaxmost_words(semantic_word_df)
        
        self.features_wordlevel["bsmVAD_glob_max_word_v"] = semantic_word_df.at[words['highest_valence_word'], "V.Mean.Sum"]
        self.features_wordlevel["bsmVAD_glob_max_word_a"] = semantic_word_df.at[words['highest_arousal_word'], "A.Mean.Sum"]
        self.features_wordlevel["bsmVAD_glob_max_word_d"] = semantic_word_df.at[words['highest_dominance_word'], "D.Mean.Sum"]

        self.features_wordlevel["bsmVAD_glob_min_word_v"] = semantic_word_df.at[words['lowest_valence_word'], "V.Mean.Sum"]
        self.features_wordlevel["bsmVAD_glob_min_word_a"] = semantic_word_df.at[words['lowest_arousal_word'], "A.Mean.Sum"]
        self.features_wordlevel["bsmVAD_glob_min_word_d"] = semantic_word_df.at[words['lowest_dominance_word'], "D.Mean.Sum"]
        
        self.features_wordlevel["bsmVAD_glob_most_word_v"] = semantic_word_df.at[words['most_occuring_word'], "V.Mean.Sum"]
        self.features_wordlevel["bsmVAD_glob_most_word_a"] = semantic_word_df.at[words['most_occuring_word'], "A.Mean.Sum"]
        self.features_wordlevel["bsmVAD_glob_most_word_d"] = semantic_word_df.at[words['most_occuring_word'], "D.Mean.Sum"]
        self.features_wordlevel["bsmVAD_glob_most_word_count"] = semantic_word_df.at[words['most_occuring_word'], "Count"]

    def _get_minmaxmost_words(self, semantic_word_df) -> dict:
        return {
            "highest_valence_word": semantic_word_df['V.Mean.Sum'].idxmax(),
            "highest_arousal_word": semantic_word_df['A.Mean.Sum'].idxmax(),
            "highest_dominance_word": semantic_word_df['D.Mean.Sum'].idxmax(),
            "lowest_valence_word": semantic_word_df['V.Mean.Sum'].idxmin(), 
            "lowest_arousal_word": semantic_word_df['A.Mean.Sum'].idxmin(), 
            "lowest_dominance_word": semantic_word_df['D.Mean.Sum'].idxmin(),
            "most_occuring_word": semantic_word_df['Count'].idxmax()
        }

