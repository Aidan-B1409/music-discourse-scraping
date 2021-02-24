import pandas as pd 


from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

def iteraffects():
    for datatype in ['positive', 'negative', 'anger', 'anticipation', 'disgust',
     'fear', 'joy', 'sadness', 'surprise', 'trust']:
        yield datatype

def get_glob_headers() -> dict:
    dict = {}
    for datatype in iteraffects():
        for uniq_status in ['uniq_', '']:
            dict[f"EmoLex_glob_{datatype}_{uniq_status}mean"] = 0.0
        dict[f"EmoLex_glob_{datatype}_most_freq_mean"] = 0.0
    dict['EmoLex_glob_ratio'] = 0.0
    dict['EmoLex_glob_ratio_uniq'] = 0.0
    return dict

def get_commentlevel_headers() -> dict:
    dict = {}
    for value in ['mean', 'std']:
        for datatype in iteraffects():
            for uniq_status in ['_uniq', '']:
                dict[f'EmoLex_{datatype}{uniq_status}_means_{value}'] = 0.0
            dict[f'EmoLex_{datatype}_most_freq_means_{value}'] = 0.0
        dict[f'EmoLex_ratio_{value}'] = 0.0
        dict[f'EmoLex_ratio_uniq_{value}'] = 0.0
    return dict

def get_header():
    header = {}
    header.update(get_glob_headers())
    header.update(get_commentlevel_headers())
    return header

class EmoLex_wlist:

    def __init__(self, emolex_df) -> None:
        columns = []
        for datatype in iteraffects():
            for uniq_status in ['uniq_', '']:
                columns.append(f'{datatype}_{uniq_status}means')
            columns.append(f'{datatype}_most_freq_means')
        columns.append('ratio')
        columns.append('ratio_uniq')

        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.emolex_df = emolex_df
        self.features_wordlevel = get_glob_headers()
        self.features_commentlevel = get_commentlevel_headers()


    def wordlevel_analysis(self, song_df, glob_df) -> dict:
        root = "EmoLex_glob"
        if(len(song_df) > 0):
            found_word_df = unsquished_intersection(song_df, self.emolex_df)
            uniq_found_word_df = glob_intersection(glob_df, self.emolex_df)
            sentiment_counts = {"positive": 0, "negative":0}
            uniq_sentiment_counts = {"positive": 0, "negative":0}

            for datatype in iteraffects():
                sentiment_matched_words = self._get_affect_subset(datatype, found_word_df)
                uniq_sentiment_matched_words = self._get_affect_subset(datatype, uniq_found_word_df)
                if(len(uniq_sentiment_matched_words) > 0):
                    self.features_wordlevel[f'{root}_{datatype}_mean'] = len(sentiment_matched_words) / len(found_word_df)
                    self.features_wordlevel[f'{root}_{datatype}_uniq_mean'] = len(uniq_sentiment_matched_words) / len(uniq_found_word_df)

                    sentiment_counts[datatype] = len(sentiment_matched_words)
                    uniq_sentiment_counts[datatype] = len(uniq_sentiment_matched_words)
                    self.features_wordlevel[f'{root}_{datatype}_most_freq_mean'] = self._getfreqmean(uniq_sentiment_matched_words, datatype)
            
            if(uniq_sentiment_counts['negative'] > 0):
                self.features_wordlevel[f'{root}_ratio'] = sentiment_counts['positive'] / sentiment_counts['negative']
                self.features_wordlevel[f'{root}_ratio_uniq'] = uniq_sentiment_counts['positive'] / uniq_sentiment_counts['negative']
        
        return self.features_wordlevel

    def process_comment(self, index, comment):
        comment_list = clean_comment(comment)

        unique_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])

        found_word_df = pd.merge(words_df, self.emolex_df, on='Word')
        uniq_found_word_df = glob_intersection(unique_words_df, self.emolex_df)

        sentiment_counts = {"positive": 0, "negative":0}
        uniq_sentiment_counts = {"positive": 0, "negative":0}

        if(len(uniq_found_word_df) > 0):
            for datatype in iteraffects():
                sentiment_matched_words = self._get_affect_subset(datatype, found_word_df)
                uniq_sentiment_matched_words = self._get_affect_subset(datatype, uniq_found_word_df)

                self.comment_analysis_df.at[index, f'{datatype}_means'] = len(sentiment_matched_words) / len(found_word_df)
                self.comment_analysis_df.at[index, f'{datatype}_uniq_means'] = len(uniq_sentiment_matched_words) / len(uniq_found_word_df)

                if(len(uniq_sentiment_matched_words) > 0):
                    self.comment_analysis_df.at[index, f'{datatype}_most_freq_means'] = self._getfreqmean(uniq_sentiment_matched_words, datatype)
                if(datatype == 'positive' or datatype == 'negative'):
                    sentiment_counts[datatype] = len(sentiment_matched_words)
                    uniq_sentiment_counts[datatype] = len(uniq_sentiment_matched_words)
            
            if(uniq_sentiment_counts['negative'] > 0):
                self.comment_analysis_df.at[index, 'ratio'] = sentiment_counts['positive'] / sentiment_counts['negative']
                self.comment_analysis_df.at[index, 'ratio_uniq'] = uniq_sentiment_counts['positive'] / uniq_sentiment_counts['negative']
    
    def analyze_comments(self) -> dict:
        for col in self.comment_analysis_df:
            self._write_mean_std(self.features_commentlevel, f'EmoLex_{col}', col, self.comment_analysis_df)
        return self.features_commentlevel

    def _getfreqmean(self, df, key):
        return df.at[self._findmax(key, df), 'Count'] / len(df)

    def _findmax(self, key, df):
        result = df['Count'].idxmax()
        while (df.at[result, 'Emotion'] != key):
            df = df.drop(result)
            result = df['Count'].idxmax()
        return result

    def _get_affect_subset(self, affect_key, emolex_df) -> pd.DataFrame:
        emolex_affect_subset = emolex_df[(emolex_df['Emotion'] == affect_key)].copy()
        indexes = emolex_affect_subset[emolex_affect_subset['Association'] == 0].index
        emolex_affect_subset.drop(indexes, inplace=True)
        return emolex_affect_subset

    def _write_mean_std(self, dict, write_key, df_key, df):
        series = df[df_key].dropna()
        data = get_mean_std(series, len(series))
        dict[f"{write_key}_mean"] = data[0]
        dict[f"{write_key}_std"] = data[1]

