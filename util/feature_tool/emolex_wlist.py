import pandas as pd 
import numpy as np

from pandas import Series
from pandas.core.algorithms import unique
from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

class EmoLex_wlist:

    #TODO - Come back to make emotion matrix

    def __init__(self, wlist_path: str) -> None:
        columns = ['positive_means', 'negative_means', 'anger_means', 'anticipation_means', 'disgust_means', 
                'fear_means', 'joy_means','sadness_means', 'surprise_means', 'trust_means', 'positive_uniq_means',
                'negative_uniq_means', 'anger_uniq_means', 'anticipation_uniq_means', 'disgust_uniq_means',
                'fear_uniq_means', 'joy_uniq_means','sadness_uniq_means', 'surprise_uniq_means', 'trust_uniq_means',
                'mostfreq_word_positive_means', 'mostfreq_word_negative_means', 'mostfreq_word_anger_means',
                'mostfreq_word_anticipation_means', 'mostfreq_word_disgust_means', 'mostfreq_word_fear_means',
                'mostfreq_word_joy_means', 'mostfreq_word_sadness_means', 'mostfreq_word_surprise_means',
                'mostfreq_word_trust_means', 'ratio', 'ratio_uniq']
        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.emolex_df = pd.read_csv(wlist_path,  names=['Word','Emotion','Association'], skiprows=1, sep='\t')

        self.features_wordlevel = {
        'EmoLex_glob_positive_mean': 0.0, 'EmoLex_glob_positive_mean_uniq': 0.0,
        'EmoLex_glob_negative_mean': 0.0, 'EmoLex_glob_negative_mean_uniq': 0.0,
        'EmoLex_glob_anger_mean': 0.0, 'EmoLex_glob_anger_mean_uniq': 0.0,
        'EmoLex_glob_anticipation_mean': 0.0, 'EmoLex_glob_anticipation_mean_uniq': 0.0,
        'EmoLex_glob_disgust_mean': 0.0, 'EmoLex_glob_disgust_mean_uniq': 0.0,
        'EmoLex_glob_fear_mean': 0.0, 'EmoLex_glob_fear_mean_uniq': 0.0,
        'EmoLex_glob_joy_mean': 0.0, 'EmoLex_glob_joy_mean_uniq': 0.0,
        'EmoLex_glob_sadness_mean': 0.0, 'EmoLex_glob_sadness_mean_uniq': 0.0,
        'EmoLex_glob_surprise_mean': 0.0, 'EmoLex_glob_surprise_mean_uniq': 0.0,
        'EmoLex_glob_trust_mean': 0.0, 'EmoLex_glob_trust_mean_uniq': 0.0,

        'EmoLex_glob_most_freq_positive_word_mean': 0.0,
        'EmoLex_glob_most_freq_negative_word_mean': 0.0,
        'EmoLex_glob_most_freq_anger_word_mean': 0.0,
        'EmoLex_glob_most_freq_anticipation_word_mean': 0.0,
        'EmoLex_glob_most_freq_disgust_word_mean': 0.0,
        'EmoLex_glob_most_freq_fear_word_mean': 0.0,
        'EmoLex_glob_most_freq_joy_word_mean': 0.0,
        'EmoLex_glob_most_freq_sadness_word_mean': 0.0,
        'EmoLex_glob_most_freq_surprise_word_mean': 0.0,
        'EmoLex_glob_most_freq_trust_word_mean': 0.0,

        'EmoLex_glob_posneg_ratio': 0.0, 'EmoLex_glob_posneg_ratio_uniq': 0.0
        }

        self.features_commentlevel = {
        'EmoLex_positive_means_mean': 0.0, 'EmoLex_positive_means_std': 0.0,
        'EmoLex_positive_means_uniq_mean': 0.0, 'EmoLex_positive_means_uniq_std': 0.0,
        'EmoLex_negative_means_mean': 0.0, 'EmoLex_negative_means_std': 0.0,
        'EmoLex_negative_means_uniq_mean': 0.0, 'EmoLex_negative_means_uniq_std': 0.0,
        'EmoLex_anger_means_mean': 0.0, 'EmoLex_anger_means_std': 0.0, 
        'EmoLex_anger_means_uniq_mean': 0.0, 'EmoLex_anger_means_uniq_std': 0.0,
        'EmoLex_anticipation_means_mean': 0.0, 'EmoLex_anticipation_means_std': 0.0,
        'EmoLex_anticipation_means_uniq_mean': 0.0, 'EmoLex_anticipation_means_uniq_std': 0.0,
        'EmoLex_disgust_means_mean': 0.0, 'EmoLex_disgust_means_std': 0.0,
        'EmoLex_disgust_means_uniq_mean': 0.0, 'EmoLex_disgust_means_uniq_std': 0.0,
        'EmoLex_fear_means_mean': 0.0, 'EmoLex_fear_means_std': 0.0,
        'EmoLex_fear_means_uniq_mean': 0.0, 'EmoLex_fear_means_uniq_std': 0.0,
        'EmoLex_joy_means_mean': 0.0, 'EmoLex_joy_means_std': 0.0, 
        'EmoLex_joy_means_uniq_mean': 0.0, 'EmoLex_joy_means_uniq_std': 0.0,
        'EmoLex_sadness_means_mean': 0.0, 'EmoLex_sadness_means_std': 0.0,
        'EmoLex_sadness_means_uniq_mean': 0.0, 'EmoLex_sadness_means_uniq_std': 0.0,
        'EmoLex_surprise_means_mean': 0.0, 'EmoLex_surprise_means_std': 0.0, 
        'EmoLex_surprise_means_uniq_mean': 0.0, 'EmoLex_surprise_means_uniq_std': 0.0,
        'EmoLex_trust_means_mean': 0.0, 'EmoLex_trust_means_std': 0.0, 
        'EmoLex_trust_means_uniq_mean': 0.0, 'EmoLex_trust_means_uniq_std': 0.0,

        'EmoLex_most_freq_positive_word_means_mean': 0.0, 'EmoLex_most_freq_positive_word_means_std': 0.0,
        'EmoLex_most_freq_negative_word_means_mean': 0.0, 'EmoLex_most_freq_negative_word_means_std': 0.0,
        'EmoLex_most_freq_anger_word_means_mean': 0.0, 'EmoLex_most_freq_anger_word_means_std': 0.0,
        'EmoLex_most_freq_anticipation_word_means_mean': 0.0, 'EmoLex_most_freq_anticipation_word_means_std': 0.0,
        'EmoLex_most_freq_disgust_word_means_mean': 0.0, 'EmoLex_most_freq_disgust_word_means_std': 0.0,
        'EmoLex_most_freq_fear_word_means_mean': 0.0, 'EmoLex_most_freq_fear_word_means_std': 0.0,
        'EmoLex_most_freq_joy_word_means_mean': 0.0, 'EmoLex_most_freq_joy_word_means_std': 0.0,
        'EmoLex_most_freq_sadness_word_means_mean': 0.0, 'EmoLex_most_freq_sadness_word_means_std': 0.0,
        'EmoLex_most_freq_surprise_word_means_mean': 0.0, 'EmoLex_most_freq_surprise_word_means_std': 0.0,
        'EmoLex_most_freq_trust_word_means_mean': 0.0, 'EmoLex_most_freq_trust_word_means_std': 0.0,

        'EmoLex_posneg_ratio_mean': 0.0, 'EmoLex_posneg_ratio_std': 0.0, 
        'EmoLex_posneg_ratio_uniq_mean': 0.0, 'EmoLex_posneg_ratio_uniq_std': 0.0, 
        }


    def wordlevel_analysis(self, song_df, glob_df) -> dict:
        if(len(song_df) > 0):

            semantic_word_df = unsquished_intersection(song_df, self.emolex_df)
            indexes = semantic_word_df[semantic_word_df['Association'] == 0].index
            semantic_word_df.drop(indexes, inplace=True)

            uniq_semantic_word_df = glob_intersection(glob_df, self.emolex_df)
            indexes_uniq = uniq_semantic_word_df[uniq_semantic_word_df['Association'] == 0].index
            uniq_semantic_word_df.drop(indexes_uniq, inplace=True)

            positive_df = semantic_word_df[(semantic_word_df['Emotion'] == 'positive')]
            negative_df = semantic_word_df[(semantic_word_df['Emotion'] == 'negative')]
            anger_df = semantic_word_df[(semantic_word_df['Emotion'] == 'anger')]
            anticipation_df = semantic_word_df[(semantic_word_df['Emotion'] == 'anticipation')]
            disgust_df = semantic_word_df[(semantic_word_df['Emotion'] == 'disgust')]
            fear_df = semantic_word_df[(semantic_word_df['Emotion'] == 'fear')]
            joy_df = semantic_word_df[(semantic_word_df['Emotion'] == 'joy')]
            sadness_df = semantic_word_df[(semantic_word_df['Emotion'] == 'sadness')]
            surprise_df = semantic_word_df[(semantic_word_df['Emotion'] == 'surprise')]
            trust_df = semantic_word_df[(semantic_word_df['Emotion'] == 'trust')]

            uniq_positive_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'positive')]
            uniq_negative_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'negative')]
            uniq_anger_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'anger')]
            uniq_anticipation_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'anticipation')]
            uniq_disgust_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'disgust')]
            uniq_fear_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'fear')]
            uniq_joy_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'joy')]
            uniq_sadness_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'sadness')]
            uniq_surprise_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'surprise')]
            uniq_trust_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'trust')]


            if(len(semantic_word_df) > 0):
                self.features_wordlevel['EmoLex_glob_positive_mean'] = len(positive_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_negative_mean'] = len(negative_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_anger_mean'] = len(anger_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_anticipation_mean'] = len(anticipation_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_disgust_mean'] = len(disgust_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_fear_mean'] = len(fear_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_joy_mean'] = len(joy_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_sadness_mean'] = len(sadness_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_surprise_mean'] = len(surprise_df) / len(semantic_word_df)
                self.features_wordlevel['EmoLex_glob_trust_mean'] = len(trust_df) / len(semantic_word_df)

                if(len(negative_df) > 0):
                    self.features_wordlevel['EmoLex_glob_posneg_ratio'] = len(positive_df) / len(negative_df)


            if(len(uniq_semantic_word_df)):
                self.features_wordlevel['EmoLex_glob_positive_mean_uniq'] = len(uniq_positive_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_negative_mean_uniq'] = len(uniq_negative_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_anger_mean_uniq'] = len(uniq_anger_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_anticipation_mean_uniq'] = len(uniq_anticipation_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_disgust_mean_uniq'] = len(uniq_disgust_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_fear_mean_uniq'] = len(uniq_fear_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_joy_mean_uniq'] = len(uniq_joy_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_sadness_mean_uniq'] = len(uniq_sadness_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_surprise_mean_uniq'] = len(uniq_surprise_df) / len(uniq_semantic_word_df)
                self.features_wordlevel['EmoLex_glob_trust_mean_uniq'] = len(uniq_trust_df) / len(uniq_semantic_word_df)

                self.features_wordlevel['EmoLex_glob_most_freq_positive_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'positive') if len(uniq_positive_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_negative_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'negative') if len(uniq_negative_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_anger_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'anger') if len(uniq_anger_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_anticipation_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'anticipation') if len(uniq_anticipation_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_disgust_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'disgust') if len(uniq_disgust_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_fear_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'fear') if len(uniq_fear_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_joy_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'joy') if len(uniq_joy_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_sadness_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'sadness') if len(uniq_sadness_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_surprise_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'surprise') if len(uniq_surprise_df) > 0 else 0
                self.features_wordlevel['EmoLex_glob_most_freq_trust_word_mean'] = self._getfreqmean(uniq_semantic_word_df, 'trust') if len(uniq_trust_df) > 0 else 0

                if(len(uniq_negative_df) > 0):
                    self.features_wordlevel['EmoLex_glob_posneg_ratio_uniq'] = len(uniq_positive_df) / len(uniq_negative_df)

        return self.features_wordlevel

    def _getfreqmean(self, df, key):
        return df.at[self._findmax(key, df), 'Count'] / len(df)

    def _findmax(self, key, df):
      result = df['Count'].idxmax()
      while (df.at[result, 'Emotion'] != key):
        df = df.drop(result)
        result = df['Count'].idxmax()
      return result

    def _semratio(self, series1, series2):
        return len(series1) / len(series2)

    def process_comment(self, index, comment):
        # clean the string
        comment_list = clean_comment(comment)

        unique_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])

        semantic_word_df = pd.merge(words_df, self.emolex_df)
        indexes = semantic_word_df[semantic_word_df['Association'] == 0].index
        semantic_word_df.drop(indexes, inplace=True)

        uniq_semantic_word_df = glob_intersection(unique_words_df, self.emolex_df)
        indexes_uniq = uniq_semantic_word_df[uniq_semantic_word_df['Association'] == 0].index
        uniq_semantic_word_df.drop(indexes_uniq, inplace=True)


        positive_df = semantic_word_df[(semantic_word_df['Emotion'] == 'positive')]
        negative_df = semantic_word_df[(semantic_word_df['Emotion'] == 'negative')]
        anger_df = semantic_word_df[(semantic_word_df['Emotion'] == 'anger')]
        anticipation_df = semantic_word_df[(semantic_word_df['Emotion'] == 'anticipation')]
        disgust_df = semantic_word_df[(semantic_word_df['Emotion'] == 'disgust')]
        fear_df = semantic_word_df[(semantic_word_df['Emotion'] == 'fear')]
        joy_df = semantic_word_df[(semantic_word_df['Emotion'] == 'joy')]
        sadness_df = semantic_word_df[(semantic_word_df['Emotion'] == 'sadness')]
        surprise_df = semantic_word_df[(semantic_word_df['Emotion'] == 'surprise')]
        trust_df = semantic_word_df[(semantic_word_df['Emotion'] == 'trust')]

        uniq_positive_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'positive')]
        uniq_negative_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'negative')]
        uniq_anger_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'anger')]
        uniq_anticipation_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'anticipation')]
        uniq_disgust_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'disgust')]
        uniq_fear_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'fear')]
        uniq_joy_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'joy')]
        uniq_sadness_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'sadness')]
        uniq_surprise_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'surprise')]
        uniq_trust_df = uniq_semantic_word_df[(uniq_semantic_word_df['Emotion'] == 'trust')]

        if len(semantic_word_df) > 0:
            self.comment_analysis_df.at[index, 'positive_means'] = len(positive_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'negative_means'] = len(negative_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'anger_means'] = len(anger_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'anticipation_means'] = len(anticipation_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'disgust_means'] = len(disgust_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'fear_means'] = len(fear_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'joy_means'] = len(joy_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'sadness_means'] = len(sadness_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'surprise_means'] = len(surprise_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'trust_means'] = len(trust_df) / len(semantic_word_df)

            if(len(negative_df) > 0):
                self.comment_analysis_df.at[index, 'ratio'] = len(positive_df) / len(negative_df)
            

        if(len(uniq_semantic_word_df) > 0):
            self.comment_analysis_df.at[index, 'positive_uniq_means'] = len(uniq_positive_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'negative_uniq_means'] = len(uniq_negative_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'anger_uniq_means'] = len(uniq_anger_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'anticipation_uniq_means'] = len(uniq_anticipation_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'disgust_uniq_means'] = len(uniq_disgust_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'fear_uniq_means'] = len(uniq_fear_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'joy_uniq_means'] = len(uniq_joy_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'sadness_uniq_means'] = len(uniq_sadness_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'surprise_uniq_means'] = len(uniq_surprise_df) / len(semantic_word_df)
            self.comment_analysis_df.at[index, 'trust_uniq_means'] = len(uniq_trust_df) / len(semantic_word_df)
            
            
            self.comment_analysis_df.at[index, 'mostfreq_word_positive_means'] = self._getfreqmean(uniq_semantic_word_df, 'positive') if len(uniq_positive_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_negative_means'] = self._getfreqmean(uniq_semantic_word_df, 'negative') if len(uniq_negative_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_anger_means'] = self._getfreqmean(uniq_semantic_word_df, 'anger') if len(uniq_anger_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_anticipation_means'] = self._getfreqmean(uniq_semantic_word_df, 'anticipation') if len(uniq_anticipation_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_disgust_means'] = self._getfreqmean(uniq_semantic_word_df, 'disgust') if len(uniq_disgust_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_fear_means'] = self._getfreqmean(uniq_semantic_word_df, 'fear') if len(uniq_fear_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_joy_means'] = self._getfreqmean(uniq_semantic_word_df, 'joy') if len(uniq_joy_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_sadness_means'] = self._getfreqmean(uniq_semantic_word_df, 'sadness') if len(uniq_sadness_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_surprise_means'] = self._getfreqmean(uniq_semantic_word_df, 'surprise') if len(uniq_surprise_df) > 0 else np.NaN
            self.comment_analysis_df.at[index, 'mostfreq_word_trust_means'] = self._getfreqmean(uniq_semantic_word_df, 'trust') if len(uniq_trust_df) > 0 else np.NaN

            if(len(uniq_negative_df) > 0):
                self.comment_analysis_df.at[index, 'ratio_uniq'] = len(uniq_positive_df) / len(uniq_negative_df)

    def analyze_comments(self) -> dict:
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_positive_means_mean', 'EmoLex_positive_means_std', self.comment_analysis_df['positive_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_negative_means_mean', 'EmoLex_negative_means_std', self.comment_analysis_df['negative_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_anger_means_mean', 'EmoLex_anger_means_std', self.comment_analysis_df['anger_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_anticipation_means_mean', 'EmoLex_anticipation_means_std', self.comment_analysis_df['anticipation_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_disgust_means_mean', 'EmoLex_disgust_means_std', self.comment_analysis_df['disgust_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_fear_means_mean', 'EmoLex_fear_means_std', self.comment_analysis_df['fear_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_joy_means_mean', 'EmoLex_joy_means_std', self.comment_analysis_df['joy_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_sadness_means_mean', 'EmoLex_sadness_means_std', self.comment_analysis_df['sadness_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_surprise_means_mean', 'EmoLex_surprise_means_std', self.comment_analysis_df['surprise_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_trust_means_mean', 'EmoLex_trust_means_std', self.comment_analysis_df['trust_means'])

        lwrite_mean_std(self.features_commentlevel, 'EmoLex_positive_means_uniq_mean', 'EmoLex_positive_means_uniq_std', self.comment_analysis_df['positive_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_negative_means_uniq_mean', 'EmoLex_negative_means_uniq_std', self.comment_analysis_df['negative_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_anger_means_uniq_mean', 'EmoLex_anger_means_uniq_std', self.comment_analysis_df['anger_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_anticipation_means_uniq_mean', 'EmoLex_anticipation_means_uniq_std', self.comment_analysis_df['anticipation_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_disgust_means_uniq_mean', 'EmoLex_disgust_means_uniq_std', self.comment_analysis_df['disgust_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_fear_means_uniq_mean', 'EmoLex_fear_means_uniq_std', self.comment_analysis_df['fear_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_joy_means_uniq_mean', 'EmoLex_joy_means_uniq_std', self.comment_analysis_df['joy_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_sadness_means_uniq_mean', 'EmoLex_sadness_means_uniq_std', self.comment_analysis_df['sadness_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_surprise_means_uniq_mean', 'EmoLex_surprise_means_uniq_std', self.comment_analysis_df['surprise_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_trust_means_uniq_mean', 'EmoLex_trust_means_uniq_std', self.comment_analysis_df['trust_uniq_means'])


        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_positive_word_means_mean', 'EmoLex_most_freq_positive_word_means_std', self.comment_analysis_df['mostfreq_word_positive_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_negative_word_means_mean', 'EmoLex_most_freq_negative_word_means_std', self.comment_analysis_df['mostfreq_word_negative_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_anger_word_means_mean', 'EmoLex_most_freq_anger_word_means_std', self.comment_analysis_df['mostfreq_word_anger_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_anticipation_word_means_mean', 'EmoLex_most_freq_anticipation_word_means_std', self.comment_analysis_df['mostfreq_word_anticipation_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_disgust_word_means_mean', 'EmoLex_most_freq_disgust_word_means_std', self.comment_analysis_df['mostfreq_word_disgust_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_fear_word_means_mean', 'EmoLex_most_freq_fear_word_means_std', self.comment_analysis_df['mostfreq_word_fear_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_joy_word_means_mean', 'EmoLex_most_freq_joy_word_means_std', self.comment_analysis_df['mostfreq_word_joy_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_sadness_word_means_mean', 'EmoLex_most_freq_sadness_word_means_std', self.comment_analysis_df['mostfreq_word_sadness_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_surprise_word_means_mean', 'EmoLex_most_freq_surprise_word_means_std', self.comment_analysis_df['mostfreq_word_surprise_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_most_freq_trust_word_means_mean', 'EmoLex_most_freq_trust_word_means_std', self.comment_analysis_df['mostfreq_word_trust_means'].dropna())


        lwrite_mean_std(self.features_commentlevel, 'EmoLex_posneg_ratio_mean', 'EmoLex_posneg_ratio_std', self.comment_analysis_df['ratio'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'EmoLex_posneg_ratio_uniq_mean', 'EmoLex_posneg_ratio_uniq_std', self.comment_analysis_df['ratio_uniq'].dropna())

        return self.features_commentlevel

