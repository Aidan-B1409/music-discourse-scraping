import pandas as pd 

from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

def iteraffects():
    for datatype in ['anger', 'fear', 'sadness', 'joy']:
        yield datatype

def get_glob_headers() -> dict:
    dict = {}
    for datatype in iteraffects():
            for analysis_type in ['mean', 'std']:
                for uniq_status in ['uniq_', '']:
                    dict[f"EmoAff_glob_{datatype}_{uniq_status}{analysis_type}"] = 0.0
    for datatype in iteraffects():
        for analysis_type in ['max_word', 'min_word', 'most_word']:
            dict[f"EmoAff_glob_{datatype}_{analysis_type}"] = 0.0
    return dict

def get_commentlevel_headers() -> dict:
    dict = {}
    for datatype in iteraffects():
        for analysis_type in ['means', 'stds']:
            for uniq_status in ['_uniq', '']:
                for value in ['mean', 'std']:
                    dict[f'EmoAff_{datatype}{uniq_status}_{analysis_type}_{value}'] = 0.0
    for datatype in iteraffects():
        for analysis_type in ['max_words', 'min_words', 'most_words']:
            for value in ['mean', 'std']:
                dict[f'EmoAff_{datatype}_{analysis_type}_{value}'] = 0.0
    return dict

def get_header():
    header = {}
    header.update(get_glob_headers())
    header.update(get_commentlevel_headers())
    return header

class EmoAff_wlist:

    def __init__(self, wlist_path: str) -> None:
        columns = []
        for datatype in iteraffects():
            for analysis in ['means', 'stds']:
                for uniq_status in ['_uniq', '']:
                    columns.append(f"{datatype}{uniq_status}_{analysis}")
        for datatype in iteraffects():
            for analysis in ['max_words', 'min_words', 'most_words']:
                columns.append(f"{datatype}_{analysis}")

        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.emoaff_df = pd.read_csv(wlist_path, names=['Word','Score','Affect'], skiprows=1, sep='\t', index_col=False)

        self.features_wordlevel = get_glob_headers()
        self.features_commentlevel = get_commentlevel_headers()

    def wordlevel_analysis(self, song_df: pd.DataFrame, glob_df: pd.DataFrame) -> dict:
        root = "EmoAff_glob"
        if(len(song_df) > 0):
            found_word_df = unsquished_intersection(song_df, self.emoaff_df)
            uniq_found_word_df = glob_intersection(glob_df, self.emoaff_df)

            for datatype in iteraffects():
                sentiment_matched_words = self._get_affect_subset(datatype, found_word_df)
                uniq_sentiment_matched_words = self._get_affect_subset(datatype, uniq_found_word_df)
                if(len(uniq_sentiment_matched_words) > 0):
                    self._write_mean_std(self.features_wordlevel, f"{root}_{datatype}", sentiment_matched_words)
                    self._write_mean_std(self.features_wordlevel, f"{root}_{datatype}_uniq", uniq_sentiment_matched_words)
                    
                    self._write_min(self.features_wordlevel, uniq_sentiment_matched_words, f"{root}_{datatype}")
                    self._write_max(self.features_wordlevel, uniq_sentiment_matched_words, f"{root}_{datatype}")
                    self._write_most(self.features_wordlevel, uniq_sentiment_matched_words, f"{root}_{datatype}")

        return self.features_wordlevel

    def process_comment(self, index, comment):
        comment_list = clean_comment(comment)

        unique_words_df = make_word_df(comment_list)
        words_df = pd.DataFrame(comment_list, columns=['Word'])

        found_word_df = pd.merge(words_df, self.emoaff_df, on='Word')
        uniq_found_word_df = glob_intersection(unique_words_df, self.emoaff_df)

        if(len(uniq_found_word_df) > 0):
            for datatype in iteraffects():
                sentiment_matched_words = self._get_affect_subset(datatype, found_word_df)
                uniq_sentiment_matched_words = self._get_affect_subset(datatype, uniq_found_word_df)

                data = get_mean_std(sentiment_matched_words['Score'], len(found_word_df))
                self.comment_analysis_df.at[index, f"{datatype}_means"] = data[0]
                self.comment_analysis_df.at[index, f"{datatype}_stds"] = data[1]

                uniq_data = get_mean_std(uniq_sentiment_matched_words['Score'], len(uniq_found_word_df))
                self.comment_analysis_df.at[index, f"{datatype}_uniq_means"] = uniq_data[0]
                self.comment_analysis_df.at[index, f"{datatype}_uniq_stds"] = uniq_data[1]

                if(len(uniq_sentiment_matched_words) > 0):
                    minmaxmost = self._get_minmaxmost_words(uniq_sentiment_matched_words)
                    self.comment_analysis_df.at[index, f"{datatype}_max_words"] = minmaxmost['max_word_score']
                    self.comment_analysis_df.at[index, f"{datatype}_min_words"] = minmaxmost['min_word_score']
                    self.comment_analysis_df.at[index, f"{datatype}_most_words"] = minmaxmost['most_word_score']


    def analyze_comments(self) -> dict:
        for col in self.comment_analysis_df:
            self._lwrite_mean_std(self.features_commentlevel, f'EmoAff_{col}', col, self.comment_analysis_df)
        return self.features_commentlevel



    def _write_min(self, dict, df, write_key):
        dict[f"{write_key}_min_word"] = df.at[df['Score'].idxmin(), 'Score']

    def _write_max(self, dict, df, write_key):
        dict[f"{write_key}_max_word"] = df.at[df['Score'].idxmax(), 'Score']

    def _write_most(self, dict, df, write_key):
        dict[f"{write_key}_most_word"] = df.at[df['Count'].idxmax(), 'Score']

    def _get_affect_subset(self, affect_key, df) -> pd.DataFrame:
        return df[(df['Affect'] == affect_key)].copy()

    def _write_mean_std(self, dict, write_key, df):
        series = df['Score'].dropna()
        data = get_mean_std(series, len(series))
        dict[f"{write_key}_mean"] = data[0]
        dict[f"{write_key}_std"] = data[1]
    
    def _lwrite_mean_std(self, dict, write_key, df_key, df):
        series = df[df_key].dropna()
        data = get_mean_std(series, len(series))
        dict[f"{write_key}_mean"] = data[0]
        dict[f"{write_key}_std"] = data[1]



    def _get_minmaxmost_words(self, df) -> dict:
        return {
            "max_word_score": df.at[df['Score'].idxmax(), 'Score'],
            "min_word_score": df.at[df['Score'].idxmin(), 'Score'],
            "most_word_score": df.at[df['Count'].idxmax(), 'Score'],
        }
