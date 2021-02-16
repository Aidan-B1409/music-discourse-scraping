import pandas as pd 
import numpy as np

from pandas import Series
from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

class MPQA_wlist:

    def __init__(self, wlist_path: str) -> None:
        columns = ['positive_means', 'positive_uniq_means', 'negative_means', 'negative_uniq_means', 
                  'most_freq_positive_word_means', 'most_freq_negative_word_means', 'ratio', 'ratio_uniq']
        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.mpqa_df = pd.read_csv(wlist_path,  names=['Word','Sentiment'], skiprows=0)

        self.features_wordlevel = {
        'mpqa_glob_positive_mean': 0.0, 'mpqa_glob_negative_mean': 0.0,
        'mpqa_glob_positive_mean_uniq': 0.0, 'mpqa_glob_negative_mean_uniq': 0.0,

        'mpqa_positive_mostfreqword_mean': 0.0, 'mpqa_negative_mostfreqword_mean': 0.0,
        'mpqa_glob_ratio': 0.0, 'mpqa_glob_ratio_uniq': 0.0
        }

        self.features_commentlevel = {
        'mpqa_positive_means_mean': 0.0, 'mpqa_positive_means_std': 0.0,
        'mpqa_positive_means_uniq_mean': 0.0, 'mpqa_positive_means_uniq_std': 0.0,
        'mpqa_negative_means_mean': 0.0, 'mpqa_negative_means_std': 0.0,
        'mpqa_negative_means_uniq_mean': 0.0, 'mpqa_negative_means_uniq_std': 0.0,

        'mpqa_mostfreq_positiveword_means_mean': 0.0, 'mpqa_mostfreq_positiveword_means_std': 0.0,
        'mpqa_mostfreq_negativeword_means_mean': 0.0, 'mpqa_mostfreq_negativeword_means_std': 0.0,
        'mpqa_ratio_mean': 0.0, 'mpqa_ratio_std': 0.0,
        'mpqa_ratio_uniq_mean': 0.0, 'mpqa_ratio_uniq_std': 0.0,
        }


    def wordlevel_analysis(self, song_df, glob_df) -> dict:
        if(len(song_df) > 0):
            allword_semantic_word_df = unsquished_intersection(song_df, self.mpqa_df)
            uniq_semantic_word_df = glob_intersection(glob_df, self.mpqa_df)

            positive_words = Series([row for row in allword_semantic_word_df.iterrows() if row[1]['Sentiment'] == 'positive'])
            negative_words = Series([row for row in allword_semantic_word_df.iterrows() if row[1]['Sentiment'] == 'negative'])
            positive_words_uniq = Series([row for row in uniq_semantic_word_df.iterrows() if row[1]['Sentiment'] == 'positive'])
            negative_words_uniq = Series([row for row in uniq_semantic_word_df.iterrows() if row[1]['Sentiment'] == 'negative'])

            if(len(allword_semantic_word_df) > 0):
                self.features_wordlevel['mpqa_glob_positive_mean'] = len(positive_words) / len(allword_semantic_word_df)
                self.features_wordlevel['mpqa_glob_negative_mean'] = len(negative_words) / len(allword_semantic_word_df)
                if(len(negative_words) > 0):
                    self.features_wordlevel['mpqa_glob_ratio'] = self._semratio(positive_words, negative_words)

            if(len(uniq_semantic_word_df)):
                self.features_wordlevel['mpqa_glob_positive_mean_uniq'] = len(positive_words_uniq) / len(uniq_semantic_word_df)
                self.features_wordlevel['mpqa_glob_negative_mean_uniq'] = len(negative_words_uniq) / len(uniq_semantic_word_df)
                if(len(negative_words_uniq) > 0):
                    self.features_wordlevel['mpqa_glob_ratio_uniq'] = self._semratio(positive_words_uniq, negative_words_uniq)
                if(len(positive_words_uniq) > 0):
                    self.features_wordlevel['mpqa_positive_mostfreqword_mean'] = self._getfreqmean(uniq_semantic_word_df, 'positive')
                if(len(negative_words_uniq) > 0):
                    self.features_wordlevel['mpqa_negative_mostfreqword_mean'] = self._getfreqmean(uniq_semantic_word_df, 'negative')
        
        return self.features_wordlevel

    def _getfreqmean(self, df, key):
        return df.at[self._findmax(key, df), 'Count'] / len(df)

    def _findmax(self, key, df):
      result = df['Count'].idxmax()
      while (df.at[result, 'Sentiment'] != key):
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

        semantic_uniq_word_df = glob_intersection(unique_words_df, self.mpqa_df)
        semantic_words_df = pd.merge(words_df, self.mpqa_df, on='Word')

        positive_words = Series([row[0] for row in semantic_words_df.iterrows() if row[1]['Sentiment'] == 'positive'])
        negative_words = Series([row[0] for row in semantic_words_df.iterrows() if row[1]['Sentiment'] == 'negative'])
        positive_words_uniq = Series([row[0] for row in semantic_uniq_word_df.iterrows() if row[1]['Sentiment'] == 'positive'])
        negative_words_uniq = Series([row[0] for row in semantic_uniq_word_df.iterrows() if row[1]['Sentiment'] == 'negative'])

        if len(semantic_words_df) > 0:
            pos_data = get_mean_std(positive_words, len(semantic_words_df))
            self.comment_analysis_df.at[index, 'positive_means'] = pos_data[0]
            self.comment_analysis_df.at[index, 'positive_stds'] = pos_data[1]

            neg_data = get_mean_std(negative_words, len(semantic_words_df))
            self.comment_analysis_df.at[index, 'negative_means'] = neg_data[0]
            self.comment_analysis_df.at[index, 'negtive_stds'] = neg_data[1]

            if len(positive_words) > 0 and len(negative_words) > 0:
                self.comment_analysis_df.at[index, 'ratio'] = self._semratio(positive_words, negative_words)

        if(len(semantic_uniq_word_df) > 0):
            pos_data = get_mean_std(positive_words, len(semantic_words_df))
            self.comment_analysis_df.at[index, 'positive_uniq_means'] = pos_data[0]
            self.comment_analysis_df.at[index, 'positive_uniq_stds'] = pos_data[1]

            neg_data = get_mean_std(negative_words, len(semantic_words_df))
            self.comment_analysis_df.at[index, 'negative_uniq_means'] = neg_data[0]
            self.comment_analysis_df.at[index, 'negative_uniq_stds'] = neg_data[1]

            if(len(positive_words_uniq) > 0):
                self.comment_analysis_df.at[index, 'most_freq_positive_word_means'] = self._getfreqmean(semantic_uniq_word_df, 'positive')
            if(len(negative_words_uniq) > 0):
                self.comment_analysis_df.at[index, 'most_freq_negative_word_means'] = self._getfreqmean(semantic_uniq_word_df, 'negative')
            if len(positive_words) > 0 and len(negative_words) > 0:
                self.comment_analysis_df.at[index, 'ratio_uniq'] = self._semratio(positive_words_uniq, negative_words_uniq)



    def analyze_comments(self) -> dict:
        lwrite_mean_std(self.features_commentlevel, 'mpqa_positive_means_mean', 'mpqa_positive_means_std', self.comment_analysis_df['positive_means'])
        lwrite_mean_std(self.features_commentlevel, 'mpqa_positive_means_uniq_mean', 'mpqa_positive_means_uniq_std', self.comment_analysis_df['positive_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'mpqa_negative_means_mean', 'mpqa_negative_means_std', self.comment_analysis_df['negative_means'])
        lwrite_mean_std(self.features_commentlevel, 'mpqa_negative_means_uniq_mean', 'mpqa_negative_means_uniq_std', self.comment_analysis_df['negative_uniq_means'])
        lwrite_mean_std(self.features_commentlevel, 'mpqa_mostfreq_positiveword_means_mean', 'mpqa_mostfreq_positiveword_means_std', self.comment_analysis_df['most_freq_positive_word_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'mpqa_mostfreq_negativeword_means_mean', 'mpqa_mostfreq_negativeword_means_std', self.comment_analysis_df['most_freq_negative_word_means'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'mpqa_ratio_mean', 'mpqa_ratio_std', self.comment_analysis_df['ratio'].dropna())
        lwrite_mean_std(self.features_commentlevel, 'mpqa_ratio_uniq_mean', 'mpqa_ratio_uniq_std', self.comment_analysis_df['ratio_uniq'].dropna())

        return self.features_commentlevel

