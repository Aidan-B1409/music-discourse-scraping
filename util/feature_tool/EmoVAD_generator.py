import pandas as pd
from string_cleaner import make_word_df
from glob_maker import make_unsquished_glob
from string_cleaner import clean_comment
from statistics import mean
from statistics import stdev

class EmoVADGenerator:

    features = {
        "EmoVAD_glob_v_mean": 0.0, "EmoVAD_glob_v_stdev": 0.0,
        "EmoVAD_glob_a_mean": 0.0, "EmoVAD_glob_a_stdev": 0.0,
        "EmoVAD_glob_d_mean": 0.0, "EmoVAD_glob_d_stdev": 0.0,
        "EmoVAD_glob_v_mean_uniq": 0.0, "EmoVAD_glob_v_stdev_uniq": 0.0,
        "EmoVAD_glob_a_mean_uniq": 0.0, "EmoVAD_glob_a_stdev_uniq": 0.0,
        "EmoVAD_glob_d_mean_uniq": 0.0, "EmoVAD_glob_d_stdev_uniq": 0.0,

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

    }

    def __init__(self, song_df, glob_df, emovad_df) -> None:
        self.song_df = song_df
        self.glob_df = glob_df
        self.emovad_df = emovad_df

    def get_features(self) -> dict:
        self.glob_features()
        self.independent_features()
        return self.features


    def std_check(self, series, mean) -> float:
        if(len(series) >= 2):
            return stdev(series, xbar = mean)
        else:
            return 0

    # Find the mean and standard deviation of a given series from a dataframe, with a length unbound from the dataframe itself
    # returns (mean, std) tuple
    def get_mean_std(self, series: pd.Series, length) -> tuple:
        if length > 0:
            mean = series.sum() / length
            std = self.std_check(series, mean)
            return (mean, std)
        return (0, 0)

    def write_mean_std(self, mean_key, std_key, data: tuple) -> None:
        self.features[mean_key] = data[0]
        self.features[std_key] = data [1]
    
    # Find the features of the word glob for a specific song 
    def glob_features(self) -> None:

        # Safety check! Dump the global dict
        for key in self.features:
            self.features[key] = 0.0

        if(len(self.song_df) > 0):
            allword_semantic_word_df = self.song_df_intersection(self.song_df, self.emovad_df)
            uniq_semantic_word_df = self.make_glob_intersection(self.glob_df, self.emovad_df)

            self.write_mean_std("EmoVAD_glob_v_mean", "EmoVAD_glob_v_stdev", self.get_mean_std(allword_semantic_word_df['Valence'], len(allword_semantic_word_df['Word'])))
            self.write_mean_std("EmoVAD_glob_a_mean", "EmoVAD_glob_a_stdev", self.get_mean_std(allword_semantic_word_df['Arousal'], len(allword_semantic_word_df['Word'])))
            self.write_mean_std("EmoVAD_glob_d_mean", "EmoVAD_glob_d_stdev", self.get_mean_std(allword_semantic_word_df['Dominance'], len(allword_semantic_word_df['Word'])))

            #VAD means on unique words only
            self.write_mean_std("EmoVAD_glob_v_mean_uniq", "EmoVAD_glob_v_stdev_uniq", self.get_mean_std(uniq_semantic_word_df['Valence'], len(uniq_semantic_word_df['Word'])))
            self.write_mean_std("EmoVAD_glob_a_mean_uniq", "EmoVAD_glob_a_stdev_uniq", self.get_mean_std(uniq_semantic_word_df['Arousal'], len(uniq_semantic_word_df['Word'])))
            self.write_mean_std("EmoVAD_glob_d_mean_uniq", "EmoVAD_glob_d_stdev_uniq", self.get_mean_std(uniq_semantic_word_df['Dominance'], len(uniq_semantic_word_df['Word'])))


    def make_glob_intersection(self, glob_df, wordlist_df) -> pd.DataFrame:
        semantic_word_df = pd.merge(glob_df, wordlist_df, on='Word')

        semantic_word_df['V_Total'] = semantic_word_df['Count'] * semantic_word_df['Valence']
        semantic_word_df['A_Total'] = semantic_word_df['Count'] * semantic_word_df['Arousal']
        semantic_word_df['D_Total'] = semantic_word_df['Count'] * semantic_word_df['Dominance']
        return semantic_word_df

    def song_df_intersection(self, song_df, wordlist_df) -> pd.DataFrame:
        semantic_wordbag_df = pd.merge(make_unsquished_glob(song_df), wordlist_df, on='Word')
        return semantic_wordbag_df


    # For each comment in the song dataframe, clean the comment string, 
    # TODO - completely rebuild function for nan safety

    def independent_features(self) -> None: 
            columns = ['valence_means', 'valence_stds', 'valence_uniq_means', 'valence_uniq_stds',
                'arousal_means', 'arousal_stds', 'arousal_uniq_means', 'arousal_uniq_stds',
                'dominance_means', 'dominance_stds', 'dominance_uniq_means', 'dominance_uniq_stds']
            
            feature_count_df = pd.DataFrame(columns=columns)
            for i, row in enumerate(self.song_df['Comment Body']):
                # clean the string
                comment_list = clean_comment(row)

                unique_words_df = make_word_df(comment_list)
                words_df = pd.DataFrame(comment_list, columns=['Word'])

                semantic_uniq_word_df = self.make_glob_intersection(unique_words_df, self.emovad_df)
                semantic_words_df = pd.merge(words_df, self.emovad_df, on='Word')

                # If there are any words which match our VAD wordlist 
                if len(unique_words_df) > 0:

                    v_data = self.get_mean_std(semantic_words_df['Valence'], len(semantic_words_df['Word']))
                    feature_count_df.at[i,'valence_means'] = v_data[0]
                    feature_count_df.at[i,'valence_stds'] = v_data[1]

                    v_data_uniq = self.get_mean_std(semantic_uniq_word_df['Valence'], len(semantic_uniq_word_df['Word']))
                    feature_count_df.at[i,'valence_uniq_means'] = v_data_uniq[0]
                    feature_count_df.at[i,'valence_uniq_stds'] = v_data_uniq[1]

                    a_data = self.get_mean_std(semantic_words_df['Arousal'], len(semantic_words_df['Word']))
                    feature_count_df.at[i,'arousal_means'] = a_data[0]
                    feature_count_df.at[i,'arousal_stds'] = a_data[1]

                    a_data_uniq = self.get_mean_std(semantic_uniq_word_df['Arousal'], len(semantic_uniq_word_df['Word']))
                    feature_count_df.at[i,'arousal_uniq_means'] = a_data_uniq[0]
                    feature_count_df.at[i,'arousal_uniq_stds'] = a_data_uniq[1]

                    d_data = self.get_mean_std(semantic_words_df['Dominance'], len(semantic_words_df['Word']))
                    feature_count_df.at[i,'dominance_means'] = d_data[0]
                    feature_count_df.at[i,'dominance_stds'] = d_data[1]

                    d_data_uniq = self.get_mean_std(semantic_uniq_word_df['Dominance'], len(semantic_uniq_word_df['Word']))
                    feature_count_df.at[i,'dominance_uniq_means'] = d_data_uniq[0]
                    feature_count_df.at[i,'dominance_uniq_stds'] = d_data_uniq[1]

            self.write_mean_std("EmoVAD_v_mean_mean", "EmoVAD_v_mean_stdev", self.get_mean_std(feature_count_df['valence_means'], len(feature_count_df['valence_means'])))
            self.write_mean_std("EmoVAD_v_std_mean", "EmoVAD_v_stdev_stdev", self.get_mean_std(feature_count_df['valence_stds'], len(feature_count_df['valence_stds'])))
            self.write_mean_std("EmoVAD_v_uniq_mean_mean", "EmoVAD_v_uniq_mean_stdev", self.get_mean_std(feature_count_df['valence_uniq_means'], len(feature_count_df['valence_uniq_means'])))
            self.write_mean_std("EmoVAD_v_uniq_std_mean", "EmoVAD_v_uniq_stdev_stdev", self.get_mean_std(feature_count_df['valence_uniq_stds'], len(feature_count_df['valence_uniq_stds'])))
            
            self.write_mean_std("EmoVAD_a_mean_mean", "EmoVAD_a_mean_stdev", self.get_mean_std(feature_count_df['arousal_means'], len(feature_count_df['arousal_means'])))
            self.write_mean_std("EmoVAD_a_std_mean", "EmoVAD_a_stdev_stdev", self.get_mean_std(feature_count_df['arousal_stds'], len(feature_count_df['arousal_stds'])))
            self.write_mean_std("EmoVAD_a_uniq_mean_mean", "EmoVAD_a_uniq_mean_stdev", self.get_mean_std(feature_count_df['arousal_uniq_means'], len(feature_count_df['arousal_uniq_means'])))
            self.write_mean_std("EmoVAD_a_uniq_std_mean", "EmoVAD_a_uniq_stdev_stdev", self.get_mean_std(feature_count_df['arousal_uniq_stds'], len(feature_count_df['arousal_uniq_stds'])))
            
            self.write_mean_std("EmoVAD_d_mean_mean", "EmoVAD_d_mean_stdev", self.get_mean_std(feature_count_df['dominance_means'], len(feature_count_df['dominance_means'])))
            self.write_mean_std("EmoVAD_d_std_mean", "EmoVAD_d_stdev_stdev", self.get_mean_std(feature_count_df['dominance_stds'], len(feature_count_df['dominance_stds'])))
            self.write_mean_std("EmoVAD_d_uniq_mean_mean", "EmoVAD_d_uniq_mean_stdev", self.get_mean_std(feature_count_df['dominance_uniq_means'], len(feature_count_df['dominance_uniq_means'])))
            self.write_mean_std("EmoVAD_d_uniq_std_mean", "EmoVAD_d_uniq_stdev_stdev", self.get_mean_std(feature_count_df['dominance_uniq_stds'], len(feature_count_df['dominance_uniq_stds'])))
