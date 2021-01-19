import numpy as np
import pandas as pd
import re
import csv
from os import listdir, walk
from pandas.core.indexes.datetimes import date_range
from tqdm import tqdm
from datetime import datetime


# Generates a list based on the comment directory
class Wordlist:

    def __init__(self, semantic_wordlist_directory, reddit_comment_directory) -> None:
        self.semantic_wordlist_directory = semantic_wordlist_directory
        self.reddit_comment_directory = reddit_comment_directory

    def filter_comments(self) -> np.ndarray:
        comment_words = self.sum_all_words()
        semantic_words = self.semantic_filter_list()
        # filtered_words = comment_words.where(comment_words in semantic_words)
        filtered_words = comment_words.isin(semantic_words)
        return comment_words[filtered_words]

    # Will iterate through every file of given directory and sum every comment body from every CSV,
    #  and then sum every word in that commend body, into one ndarray
    def sum_all_words(self) -> list():
        pbar = self.__calc_total_files()
        #word_series = pd.Series(dtype=object)
        wordlist = list()
        for subdir, dirs, files in walk(self.reddit_comment_directory):
            for file in files: 
                #Open file in pandas
                fdir = subdir + "/" + file
                comment_df = pd.read_csv(fdir, encoding='utf-8', index_col = False, engine='c')
                #for row in df
                for index, row in comment_df.iterrows():
                    for word in str(row['Comment Body']).split():
                        wordlist.append(word)
                pbar.update()
        pbar.close()
        #word_series = word_series.append(wordlist)
        word_series = pd.Series(wordlist)
        return word_series

    def __calc_total_files(self) -> tqdm:
        total = 0
        for subdir, dirs, files in walk(self.reddit_comment_directory):
            for file in files:
                total += 1
        return tqdm(total=total)


    def semantic_filter_list(self) -> pd.Series:
        filepath_bsmvad = self.semantic_wordlist_directory + '/'+'BRM-emot-submit.csv'
        filepath_emolex = self.semantic_wordlist_directory + '/'+'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
        filepath_emovad = self.semantic_wordlist_directory + '/'+'NRC-VAD-Lexicon.txt'
        filepath_emoaff = self.semantic_wordlist_directory + '/'+'NRC-AffectIntensity-Lexicon.txt'
        filepath_HSsent = self.semantic_wordlist_directory + '/'+'HS-unigrams.txt'
        filepath_anew = self.semantic_wordlist_directory + '/'+'ANEW_EnglishShortened.csv'
        filepath_mpqa = self.semantic_wordlist_directory + '/'+'MPQA_sentiment.csv'

        anew_wordarray = self.load_anew_words(filepath_anew)
        bsmvad_wordarray = self.load_bsmvad_words(filepath_bsmvad)
        emolex_wodarray = self.load_emo_words(filepath_emolex)
        emoaff_wordarray = self.load_emo_words(filepath_emoaff)
        emovad_wordarray = self.load_emo_words(filepath_emovad)
        HSsent_wordarray = self.load_HSsent_words(filepath_HSsent)
        mpqa_wordarray = self.load_mpqa_words(filepath_mpqa)

        filter_wordarray = pd.concat((anew_wordarray, bsmvad_wordarray,
        emolex_wodarray, emoaff_wordarray, emovad_wordarray, HSsent_wordarray, mpqa_wordarray))

        return filter_wordarray.drop_duplicates('first')

    # In the future, these methods may want to return the entire dataframe in order to allow for semantic-weighted wordcloud analysis.
    def load_anew_words(self, anew_path) -> pd.Series:
        return pd.read_csv(anew_path)['Word']

    def load_bsmvad_words(self, bsmvad_path) -> pd.Series:
        bsm_words = pd.read_csv(bsmvad_path, index_col=False)['Word']
        # Skip blank words 
        return bsm_words[bsm_words.notnull()]

    # Generic loader for emolex, emovad, emoaff. Will need to split into multiple methods for fetching entire dataframe, as last column or two differ per emo set
    def load_emo_words(self, emo_path) -> pd.Series:
        emolex_words = pd.read_csv(emo_path,  names=['Word','Emotion','Association'], skiprows=1, sep='\t')['Word']
        return emolex_words.drop_duplicates('first')

    def load_HSsent_words(self, HSsent_path) -> pd.Series:
        hssent_df = pd.read_csv(HSsent_path,  names=['Word','Rating','pCount','nCount'], skiprows=0, sep='\t') 
        # I'm trusting the regex logic that Donnelly wrote! 
        hssent_df['Word'] = [re.sub(r'[^\w\s\']', '', str(x)) for x in hssent_df['Word']]
        return hssent_df['Word']

    def load_mpqa_words(self, mpqa_path) -> pd.Series:
        return pd.read_csv(mpqa_path,  names=['Word','Sentiment'], skiprows=0)['Word']
      

    # Will print filtered wordlist as a csv with 1 row 
    def filtered_to_csv(self) -> None:
        wordlist = self.filter_comments()
        date_and_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        with open("filtered_reddit_comment_wordlist" + date_and_time + ".csv", 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for word in wordlist:
                csvwriter.writerow([word])

    def filtered_to_str(self) -> None:
        wordlist = self.filter_comments()
        return ''.join([word + ' ' for word in wordlist])
