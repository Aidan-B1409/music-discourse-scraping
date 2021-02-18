from numpy.core.numeric import Infinity
import pandas as pd 
import numpy as np

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
        self.emoaff_df = pd.read_csv(wlist_path,    names=['Word','Score','Affect'], skiprows=1, sep='\t')

        self.features_wordlevel = {
            'EmoAff_glob_anger_mean': 0.0, 'EmoAff_glob_anger_stdev': 0.0,
            'EmoAff_glob_fear_mean': 0.0, 'EmoAff_glob_fear_stdev': 0.0,
            'EmoAff_glob_sadness_mean': 0.0, 'EmoAff_glob_sadness_stdev': 0.0,
            'EmoAff_glob_joy_mean': 0.0, 'EmoAff_glob_joy_stdev': 0.0,
            'EmoAff_glob_anger_mean_uniq': 0.0, 'EmoAff_glob_anger_stdev_uniq': 0.0,
            'EmoAff_glob_fear_mean_uniq': 0.0, 'EmoAff_glob_fear_stdev_uniq': 0.0,
            'EmoAff_glob_sadness_mean_uniq': 0.0, 'EmoAff_glob_sadness_stdev_uniq': 0.0,
            'EmoAff_glob_joy_mean_uniq': 0.0, 'EmoAff_glob_joy_stdev_uniq': 0.0,

            'EmoAff_glob_max_word_anger': 0.0, 'EmoAff_glob_max_word_fear': 0.0, 
            'EmoAff_glob_max_word_sadness': 0.0, 'EmoAff_glob_max_word_joy': 0.0,

            'EmoAff_glob_min_word_anger': 0.0, 'EmoAff_glob_min_word_fear': 0.0, 
            'EmoAff_glob_min_word_sadness': 0.0, 'EmoAff_glob_min_word_joy': 0.0,

            'EmoAff_glob_most_word_anger': 0.0, 'EmoAff_glob_most_word_fear': 0.0, 
            'EmoAff_glob_most_word_sadness': 0.0, 'EmoAff_glob_most_word_joy': 0.0,

        }
        self.features_commentlevel = {
            'EmoAff_anger_means_mean': 0.0, 'EmoAff_anger_means_std': 0.0,
            'EmoAff_anger_stds_mean': 0.0, 'EmoAff_anger_stdev_stdev': 0.0, 
            'EmoAff_anger_uniq_means_mean': 0.0, 'EmoAff_anger_uniq_means_mean': 0.0,
            'EmoAff_anger_uniq_stds_mean': 0.0, 'EmoAff_anger_uniq_stdev_stdev': 0.0,
            
            'EmoAff_fear_means_mean': 0.0, 'EmoAff_fear_means_std': 0.0,
            'EmoAff_fear_stds_mean': 0.0, 'EmoAff_fear_stdev_stdev': 0.0, 
            'EmoAff_fear_uniq_means_mean': 0.0, 'EmoAff_fear_uniq_means_mean': 0.0,
            'EmoAff_fear_uniq_stds_mean': 0.0, 'EmoAff_fear_uniq_stdev_stdev': 0.0,

            'EmoAff_sadness_means_mean': 0.0, 'EmoAff_sadness_means_std': 0.0,
            'EmoAff_sadness_stds_mean': 0.0, 'EmoAff_sadness_stdev_stdev': 0.0, 
            'EmoAff_sadness_uniq_means_mean': 0.0, 'EmoAff_sadness_uniq_means_mean': 0.0,
            'EmoAff_sadness_uniq_stds_mean': 0.0, 'EmoAff_sadness_uniq_stdev_stdev': 0.0,

            'EmoAff_joy_means_mean': 0.0, 'EmoAff_joy_means_std': 0.0,
            'EmoAff_joy_stds_mean': 0.0, 'EmoAff_joy_stdev_stdev': 0.0, 
            'EmoAff_joy_uniq_means_mean': 0.0, 'EmoAff_joy_uniq_means_mean': 0.0,
            'EmoAff_joy_uniq_stds_mean': 0.0, 'EmoAff_joy_uniq_stdev_stdev': 0.0,

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
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_anger_mean', 'EmoAff_glob_anger_stdev', get_mean_std(anger_df, len(semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_fear_mean', 'EmoAff_glob_fear_stdev', get_mean_std(fear_df, len(semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_sadness_mean', 'EmoAff_glob_sadness_stdev', get_mean_std(sadness_df, len(semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_joy_mean', 'EmoAff_glob_joy_stdev', get_mean_std(joy_df, len(semantic_word_df)))

            if(len(uniq_semantic_word_df) > 0):
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_anger_mean_uniq', 'EmoAff_glob_anger_stdev_uniq', get_mean_std(uniq_anger_df, len(uniq_semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_fear_mean_uniq', 'EmoAff_glob_fear_stdev_uniq', get_mean_std(uniq_fear_df, len(uniq_semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_sadness_mean_uniq', 'EmoAff_glob_sadness_stdev_uniq', get_mean_std(uniq_sadness_df, len(uniq_semantic_word_df)))
                write_mean_std(self.features_wordlevel, 'EmoAff_glob_joy_mean_uniq', 'EmoAff_glob_joy_stdev_uniq', get_mean_std(uniq_joy_df, len(uniq_semantic_word_df)))

                self.features_wordlevel['EmoAff_glob_max_word_anger'] = uniq_anger_df.at[uniq_anger_df['Score'].idxmax(), 'Score']
                self.features_wordlevel['EmoAff_glob_max_word_fear'] = uniq_fear_df.at[uniq_fear_df['Score'].idxmax(), 'Score']
                self.features_wordlevel['EmoAff_glob_max_word_sadness'] = uniq_sadness_df.at[uniq_sadness_df['Score'].idxmax(), 'Score']
                self.features_wordlevel['EmoAff_glob_max_word_joy'] = uniq_joy_df.at[uniq_joy_df['Score'].idxmax(), 'Score']

                self.features_wordlevel['EmoAff_glob_min_word_anger'] = uniq_anger_df.at[uniq_anger_df['Score'].idxmin(), 'Score']
                self.features_wordlevel['EmoAff_glob_min_word_fear'] = uniq_fear_df.at[uniq_fear_df['Score'].idxmin(), 'Score']
                self.features_wordlevel['EmoAff_glob_min_word_sadness'] = uniq_sadness_df.at[uniq_sadness_df['Score'].idxmin(), 'Score']
                self.features_wordlevel['EmoAff_glob_min_word_joy'] = uniq_joy_df.at[uniq_joy_df['Score'].idxmin(), 'Score']

                self.features_wordlevel['EmoAff_glob_most_word_anger'] = uniq_anger_df.at[uniq_anger_df['Count'].idxmax(), 'Score']
                self.features_wordlevel['EmoAff_glob_most_word_fear'] = uniq_fear_df.at[uniq_fear_df['Count'].idxmax(), 'Score']
                self.features_wordlevel['EmoAff_glob_most_word_sadness'] = uniq_sadness_df.at[uniq_sadness_df['Count'].idxmax(), 'Score']
                self.features_wordlevel['EmoAff_glob_most_word_joy'] = uniq_joy_df.at[uniq_joy_df['Count'].idxmax(), 'Score']
        