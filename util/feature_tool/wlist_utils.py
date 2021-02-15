import pandas as pd

from glob_maker import make_unsquished_glob
from statistics import stdev

def write_mean_std(features_dict, mean_key, std_key, data: tuple) -> None:
    features_dict[mean_key] = data[0]
    features_dict[std_key] = data [1]

# Find the mean and standard deviation of a given series from a dataframe, with a length unbound from the dataframe itself
# returns (mean, std) tuple
# NOTE: Skipping empty values here, at the end of the pipeline. Replace with imputation later? 
def get_mean_std(series: pd.Series, length) -> tuple:
    if length > 0:
        mean = series.sum(skipna=True) / length
        std = std_check(series, mean)
        return (mean, std)
    return (0, 0)


def std_check(series, mean) -> float:
    if(len(series.dropna()) >= 2):
        return stdev(series.dropna(), xbar = mean)
    else:
        return 0


def ratio(series, threshold):
    # TODO - NANs here? 
    return sum(n > threshold for n in series) / sum(n < threshold for n in series)


# Find the intersection between a wordlist and a comment df - preserving duplicates for the sake of means. 
def unsquished_intersection(song_df, wordlist_df) -> pd.DataFrame:
    semantic_wordbag_df = pd.merge(make_unsquished_glob(song_df), wordlist_df, on='Word')
    return semantic_wordbag_df


# Find the intersecton between a wordglob (formed from comment df) and a wordlist 
def glob_intersection(glob_df, wordlist_df) -> pd.DataFrame:
    semantic_word_df = pd.merge(glob_df, wordlist_df, on='Word')
    return semantic_word_df

