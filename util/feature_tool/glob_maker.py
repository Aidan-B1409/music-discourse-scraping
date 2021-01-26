import pandas as pd
from string_cleaner import clean_comment
from string_cleaner import make_word_df


def make_glob(song_df)-> pd.DataFrame:
    # extract comments and join as one big string
    song_df['Comment Body'] = song_df['Comment Body'].astype(str)
    comment_glob = (song_df.groupby(['Query Index'])['Comment Body'].apply(lambda x: ','.join(x)).reset_index())['Comment Body'][0]
    comment_list = clean_comment(comment_glob)
    return make_word_df(comment_list)