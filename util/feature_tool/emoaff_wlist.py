from numpy.core.numeric import Infinity
import pandas as pd 

from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

class EmoAff_wlist:

    def __init__(self, wlist_path: str) -> None:
        columns = ['anger_means', 'anger_stds', 'anger_uniq_means', 'anger_uniq_stds',
            'fear_means', 'fear_stds', 'fear_uniq_means', 'fear_uniq_stds',
            'sadness_means', 'sadness_stds', 'sadness_uniq_means', 'sadness_uniq_stds', 
            'joy_means', 'joy_stds', 'joy_uniq_means', 'joy_uniq_stds', 
            'max_word_anger', 'max_word_fear', 'max_word_sadness', 'max_word_joy', 
            'min_word_anger', 'min_word_fear', 'min_word_sadness', 'min_word_joy', 
            'most_word_anger', 'most_word_fear', 'most_word_sadness', 'most_word_joy']

        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.emoaff_df = pd.read_csv(wlist_path, names=['Word','Score','Affect'], skiprows=1, sep='\t', index_col=False)

        self.features_wordlevel = {
            'EmoAff_glob_anger_mean': 0.0, 'EmoAff_glob_anger_std': 0.0,
            'EmoAff_glob_fear_mean': 0.0, 'EmoAff_glob_fear_std': 0.0,
            'EmoAff_glob_sadness_mean': 0.0, 'EmoAff_glob_sadness_std': 0.0,
            'EmoAff_glob_joy_mean': 0.0, 'EmoAff_glob_joy_std': 0.0,
            'EmoAff_glob_anger_mean_uniq': 0.0, 'EmoAff_glob_anger_std_uniq': 0.0,
            'EmoAff_glob_fear_mean_uniq': 0.0, 'EmoAff_glob_fear_std_uniq': 0.0,
            'EmoAff_glob_sadness_mean_uniq': 0.0, 'EmoAff_glob_sadness_std_uniq': 0.0,
            'EmoAff_glob_joy_mean_uniq': 0.0, 'EmoAff_glob_joy_std_uniq': 0.0,

            'EmoAff_glob_max_word_anger': 0.0, 'EmoAff_glob_max_word_fear': 0.0, 
            'EmoAff_glob_max_word_sadness': 0.0, 'EmoAff_glob_max_word_joy': 0.0,

            'EmoAff_glob_min_word_anger': 0.0, 'EmoAff_glob_min_word_fear': 0.0, 
            'EmoAff_glob_min_word_sadness': 0.0, 'EmoAff_glob_min_word_joy': 0.0,

            'EmoAff_glob_most_word_anger': 0.0, 'EmoAff_glob_most_word_fear': 0.0, 
            'EmoAff_glob_most_word_sadness': 0.0, 'EmoAff_glob_most_word_joy': 0.0,

        }
        self.features_commentlevel = {
            'EmoAff_anger_means_mean': 0.0, 'EmoAff_anger_means_std': 0.0,
            'EmoAff_anger_stds_mean': 0.0, 'EmoAff_anger_stds_std': 0.0, 
            'EmoAff_anger_uniq_means_mean': 0.0, 'EmoAff_anger_uniq_means_std': 0.0,
            'EmoAff_anger_uniq_stds_mean': 0.0, 'EmoAff_anger_uniq_stds_std': 0.0,
            
            'EmoAff_fear_means_mean': 0.0, 'EmoAff_fear_means_std': 0.0,
            'EmoAff_fear_stds_mean': 0.0, 'EmoAff_fear_stds_std': 0.0, 
            'EmoAff_fear_uniq_means_mean': 0.0, 'EmoAff_fear_uniq_means_std': 0.0,
            'EmoAff_fear_uniq_stds_mean': 0.0, 'EmoAff_fear_uniq_stds_std': 0.0,

            'EmoAff_sadness_means_mean': 0.0, 'EmoAff_sadness_means_std': 0.0,
            'EmoAff_sadness_stds_mean': 0.0, 'EmoAff_sadness_stds_std': 0.0, 
            'EmoAff_sadness_uniq_means_mean': 0.0, 'EmoAff_sadness_uniq_means_std': 0.0,
            'EmoAff_sadness_uniq_stds_mean': 0.0, 'EmoAff_sadness_uniq_stds_std': 0.0,

            'EmoAff_joy_means_mean': 0.0, 'EmoAff_joy_means_std': 0.0,
            'EmoAff_joy_stds_mean': 0.0, 'EmoAff_joy_stds_std': 0.0, 
            'EmoAff_joy_uniq_means_mean': 0.0, 'EmoAff_joy_uniq_means_std': 0.0,
            'EmoAff_joy_uniq_stds_mean': 0.0, 'EmoAff_joy_uniq_stds_std': 0.0,

            'EmoAff_anger_max_word_mean': 0.0, 'EmoAff_anger_max_word_std': 0.0, 
            'EmoAff_fear_max_word_mean': 0.0, 'EmoAff_fear_max_word_std': 0.0, 
            'EmoAff_sadness_max_word_mean': 0.0, 'EmoAff_sadness_max_word_std': 0.0, 
            'EmoAff_joy_max_word_mean': 0.0, 'EmoAff_joy_max_word_std': 0.0, 

            'EmoAff_anger_min_word_mean': 0.0, 'EmoAff_anger_min_word_std': 0.0, 
            'EmoAff_fear_min_word_mean': 0.0, 'EmoAff_fear_min_word_std': 0.0, 
            'EmoAff_sadness_min_word_mean': 0.0, 'EmoAff_sadness_min_word_std': 0.0, 
            'EmoAff_joy_min_word_mean': 0.0, 'EmoAff_joy_min_word_std': 0.0, 

            'EmoAff_anger_most_word_mean': 0.0, 'EmoAff_anger_most_word_std': 0.0, 
            'EmoAff_fear_most_word_mean': 0.0, 'EmoAff_fear_most_word_std': 0.0, 
            'EmoAff_sadness_most_word_mean': 0.0, 'EmoAff_sadness_most_word_std': 0.0, 
            'EmoAff_joy_most_word_mean': 0.0, 'EmoAff_joy_most_word_std': 0.0, 
        }

    def wordlevel_analysis(self, song_df: pd.DataFrame, glob_df: pd.DataFrame) -> dict:
        if(len(song_df) > 0):
            semantic_word_df = unsquished_intersection(song_df, self.emoaff_df)
            uniq_semantic_word_df = glob_intersection(glob_df, self.emoaff_df)

            anger_df = semantic_word_df[(semantic_word_df['Affect'] == 'anger')]
            fear_df = semantic_word_df[(semantic_word_df['Affect'] == 'fear')]
            sadness_df = semantic_word_df[(semantic_word_df['Affect'] == 'sadness')]
            joy_df = semantic_word_df[(semantic_word_df['Affect'] == 'joy')]

            uniq_anger_df = uniq_semantic_word_df[(uniq_semantic_word_df['Affect'] == 'anger')]
            uniq_fear_df = uniq_semantic_word_df[(uniq_semantic_word_df['Affect'] == 'fear')]
            uniq_sadness_df = uniq_semantic_word_df[(uniq_semantic_word_df['Affect'] == 'sadness')]
            uniq_joy_df = uniq_semantic_word_df[(uniq_semantic_word_df['Affect'] == 'joy')]

            if(len(semantic_word_df) > 0):
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_anger_mean', 'EmoAff_glob_anger_std', get_mean_std(anger_df['Score'], len(semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_fear_mean', 'EmoAff_glob_fear_std', get_mean_std(fear_df['Score'], len(semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_sadness_mean', 'EmoAff_glob_sadness_std', get_mean_std(sadness_df['Score'], len(semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_joy_mean', 'EmoAff_glob_joy_std', get_mean_std(joy_df['Score'], len(semantic_word_df)))

            if(len(uniq_semantic_word_df) > 0):
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_anger_mean_uniq', 'EmoAff_glob_anger_std_uniq', get_mean_std(uniq_anger_df['Score'], len(uniq_semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_fear_mean_uniq', 'EmoAff_glob_fear_std_uniq', get_mean_std(uniq_fear_df['Score'], len(uniq_semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_sadness_mean_uniq', 'EmoAff_glob_sadness_std_uniq', get_mean_std(uniq_sadness_df['Score'], len(uniq_semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_joy_mean_uniq', 'EmoAff_glob_joy_std_uniq', get_mean_std(uniq_joy_df['Score'], len(uniq_semantic_word_df)))

                self.features_wordlevel['EmoAff_glob_max_word_anger'] = self._get_max_element(uniq_anger_df)
                self.features_wordlevel['EmoAff_glob_max_word_fear'] = self._get_max_element(uniq_fear_df)
                self.features_wordlevel['EmoAff_glob_max_word_sadness'] = self._get_max_element(uniq_sadness_df)
                self.features_wordlevel['EmoAff_glob_max_word_joy'] = self._get_max_element(uniq_joy_df)

                self.features_wordlevel['EmoAff_glob_min_word_anger'] = self._get_min_element(uniq_anger_df)
                self.features_wordlevel['EmoAff_glob_min_word_fear'] = self._get_min_element(uniq_fear_df)
                self.features_wordlevel['EmoAff_glob_min_word_sadness'] = self._get_min_element(uniq_sadness_df)
                self.features_wordlevel['EmoAff_glob_min_word_joy'] = self._get_min_element(uniq_joy_df)

                self.features_wordlevel['EmoAff_glob_most_word_anger'] = self._get_most_element(uniq_anger_df)
                self.features_wordlevel['EmoAff_glob_most_word_fear'] = self._get_most_element(uniq_fear_df)
                self.features_wordlevel['EmoAff_glob_most_word_sadness'] = self._get_most_element(uniq_sadness_df)
                self.features_wordlevel['EmoAff_glob_most_word_joy'] = self._get_most_element(uniq_joy_df)
        return self.features_wordlevel

    def process_comment(self, index, comment):
        # clean the string
        comment_list = clean_comment(comment)

        uniq_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])
        semantic_uniq_words_df = glob_intersection(uniq_words_df, self.emoaff_df)
        semantic_words_df = pd.merge(words_df, self.emoaff_df, on='Word')

        anger_df = semantic_words_df[(semantic_words_df['Affect'] == 'anger')]
        fear_df = semantic_words_df[(semantic_words_df['Affect'] == 'fear')]
        sadness_df = semantic_words_df[(semantic_words_df['Affect'] == 'sadness')]
        joy_df = semantic_words_df[(semantic_words_df['Affect'] == 'joy')]

        uniq_anger_df = semantic_uniq_words_df[(semantic_uniq_words_df['Affect'] == 'anger')]
        uniq_fear_df = semantic_uniq_words_df[(semantic_uniq_words_df['Affect'] == 'fear')]
        uniq_sadness_df = semantic_uniq_words_df[(semantic_uniq_words_df['Affect'] == 'sadness')]
        uniq_joy_df = semantic_uniq_words_df[(semantic_uniq_words_df['Affect'] == 'joy')]

        if len(semantic_words_df) > 0:
            self._calc_mean_std(anger_df, len(semantic_words_df), 'anger', index)
            self._calc_mean_std(fear_df, len(semantic_words_df), 'fear', index)
            self._calc_mean_std(sadness_df, len(semantic_words_df), 'sadness', index)
            self._calc_mean_std(joy_df, len(semantic_words_df), 'joy', index)

        if len(semantic_uniq_words_df) > 0:
            self._calc_mean_std(uniq_anger_df, len(semantic_uniq_words_df), 'anger_uniq', index)
            self._calc_mean_std(uniq_fear_df, len(semantic_uniq_words_df), 'fear_uniq', index)
            self._calc_mean_std(uniq_sadness_df, len(semantic_uniq_words_df), 'sadness_uniq', index)
            self._calc_mean_std(uniq_joy_df, len(semantic_uniq_words_df), 'joy_uniq', index)

            self.comment_analysis_df.at[index, 'max_word_anger'] = self._get_max_element(uniq_anger_df)
            self.comment_analysis_df.at[index, 'max_word_fear'] = self._get_max_element(uniq_fear_df)
            self.comment_analysis_df.at[index, 'max_word_sadness'] = self._get_max_element(uniq_sadness_df)
            self.comment_analysis_df.at[index, 'max_word_joy'] = self._get_max_element(uniq_joy_df)

            self.comment_analysis_df.at[index, 'min_word_anger'] = self._get_min_element(uniq_anger_df)
            self.comment_analysis_df.at[index, 'min_word_fear'] = self._get_min_element(uniq_fear_df)
            self.comment_analysis_df.at[index, 'min_word_sadness'] = self._get_min_element(uniq_sadness_df)
            self.comment_analysis_df.at[index, 'min_word_joy'] = self._get_min_element(uniq_joy_df)

            self.comment_analysis_df.at[index, 'most_word_anger'] = self._get_most_element(uniq_anger_df)
            self.comment_analysis_df.at[index, 'most_word_fear'] = self._get_most_element(uniq_fear_df)
            self.comment_analysis_df.at[index, 'most_word_sadness'] = self._get_most_element(uniq_sadness_df)
            self.comment_analysis_df.at[index, 'most_word_joy'] = self._get_most_element(uniq_joy_df)

    def analyze_comments(self) -> dict:
        self._write_mean_std(self.features_commentlevel, 'EmoAff_anger_means', 'anger_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_anger_stds', 'anger_stds')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_anger_uniq_means', 'anger_uniq_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_anger_uniq_stds', 'anger_uniq_stds')
        
        self._write_mean_std(self.features_commentlevel, 'EmoAff_fear_means', 'fear_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_fear_stds', 'fear_stds')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_fear_uniq_means', 'fear_uniq_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_fear_uniq_stds', 'fear_uniq_stds')

        self._write_mean_std(self.features_commentlevel, 'EmoAff_sadness_means', 'sadness_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_sadness_stds', 'sadness_stds')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_sadness_uniq_means', 'sadness_uniq_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_sadness_uniq_stds', 'sadness_uniq_stds')

        self._write_mean_std(self.features_commentlevel, 'EmoAff_joy_means', 'joy_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_joy_stds', 'joy_stds')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_joy_uniq_means', 'joy_uniq_means')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_joy_uniq_stds', 'joy_uniq_stds')

        self._write_mean_std(self.features_commentlevel, 'EmoAff_anger_max_word', 'max_word_anger')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_fear_max_word', 'max_word_fear')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_sadness_max_word', 'max_word_sadness')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_joy_max_word', 'max_word_joy')

        self._write_mean_std(self.features_commentlevel, 'EmoAff_anger_min_word', 'min_word_anger')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_fear_min_word', 'min_word_fear')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_sadness_min_word', 'min_word_sadness')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_joy_min_word', 'min_word_joy')

        self._write_mean_std(self.features_commentlevel, 'EmoAff_anger_most_word', 'most_word_anger')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_fear_most_word', 'most_word_fear')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_sadness_most_word', 'most_word_sadness')
        self._write_mean_std(self.features_commentlevel, 'EmoAff_joy_most_word', 'most_word_joy')

        return self.features_commentlevel


    def _calc_mean_std(self, df, length, key, index):
        data = get_mean_std(df['Score'], length)
        self.comment_analysis_df.at[index, str(key) + '_means'] = data[0]
        self.comment_analysis_df.at[index, str(key) + '_stds'] = data[1]

    def _write_mean_std(self, feature_dict, writekey, readkey):
        sr = self.comment_analysis_df[readkey].dropna()
        data = get_mean_std(sr, len(sr))
        feature_dict[writekey + '_mean'] = data[0]
        feature_dict[writekey + '_std'] = data[1]

    def _get_max_element(self, df):
        if(len(df) > 0):
            return df.at[df['Score'].idxmax(), 'Score']
        return None

    def _get_min_element(self, df):
        if(len(df) > 0):
            return df.at[df['Score'].idxmin(), 'Score']
        return None

    def _get_most_element(self, df):
        if(len(df) > 0):
            return df.at[df['Count'].idxmax(), 'Score']
        return None

    
        