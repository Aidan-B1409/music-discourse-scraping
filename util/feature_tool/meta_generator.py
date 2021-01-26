import re
from statistics import mean
from statistics import stdev

from nltk.corpus.reader.util import read_wordpunct_block

class MetaGenerator:
    features = {
        "Song_ID": "",
        "Song_Artist": "", "Song_Name": "",
        "existing_valence": "", "existing_arousal": "",
        "n_comments": -1, "n_words": -1, "n_words_uniq": -1,
        "comment_length_mean": 0, "comment_length_stdev": 0
    }

    def __init__(self, song_df, glob_df) -> None:
        self.song_df = song_df
        self.glob_df = glob_df

    def get_features(self) -> dict:
        self.features["Song_ID"] = self.song_id(self.song_df)
        query = self.song_name(self.song_df)
        self.features["Song_Artist"] = query[0]
        self.features["Song_Name"] = query[1]
        self.features["existing_valence"] = self.existing_valence(self.song_df)
        self.features["existing_arousal"] = self.existing_arousal(self.song_df)

        self.drop_nan_rows()
      
        self.features["n_comments"] = self.n_comments(self.song_df)
        self.features["n_words"] = self.n_words(self.glob_df)
        self.features["n_words_uniq"]= self.n_words_uniq(self.glob_df)

        self.comment_len(self.song_df)
        return self.features


    def drop_nan_rows(self) -> None:
        #Drop all empty rows, now that we safely got the metadata
        # TODO - Do we want to throw these back to the feature extractor?   
        empty_indices = self.song_df[ self.song_df['Comment Body'] == "nan"].index
        self.song_df.drop(empty_indices, inplace=True)

        empty_glob_indices = self.glob_df[ self.glob_df['Word'] == "nan"].index
        self.glob_df.drop(empty_glob_indices, inplace=True)

    def song_id(self, song_df) -> str:
        return str(song_df.iloc[0]['Song ID'])
    
    def existing_valence(self, song_df) -> str:
        return str(song_df.iloc[0]['Valence'])
        
    def existing_arousal(self, song_df) -> str:
        return str(song_df.iloc[0]['Arousal'])

    def song_name(self, song_df) -> list:
        query = str(song_df.iloc[0]['Query'])
        artist_and_song_name = query.split('" "')
        for i in range(len(artist_and_song_name)):
            artist_and_song_name[i] = re.sub(r"title:", "", artist_and_song_name[i])
            artist_and_song_name[i] = re.sub(r'"', "", artist_and_song_name[i])
        return artist_and_song_name

    # Only useful on the glob dataframe
    # TODO - Handle off-by-one
    def n_words(self, glob_df):
        return glob_df['Count'].sum()

    # Only useful on the glob dataframe
    def n_words_uniq(self, glob_df):
        return len(glob_df)

    # It'd be cool to make this a lambda at some point
    def n_comments(self, song_df):
        return len(song_df)

    def comment_len(self, song_df):
        word_count = [len(str(x).split()) for x in song_df['Comment Body'].tolist()]
        self.features["comment_length_mean"] = 0
        self.features["comment_length_stdev"] = 0
        if len(word_count) >= 1:
            self.features["comment_length_mean"] = mean(word_count)
        if len(word_count) >= 2:
            self.features["comment_length_stdev"] = stdev(word_count)

        