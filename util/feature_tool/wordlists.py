from nltk.corpus.reader import wordlist
import pandas as pd
class WordLists():

    """
    Because all wordlists are structured differently, there's no good way
    to iteratively import and build features for all of them. 
    If adding a new wordlist, add the path and key to the dict below
    and create a new build method.
    """
    wordlists = {
        "ANEW_Extended": "BRM-emot-submit.csv",
        "ANEW_Ext_Condensed": "ANEW_EnglishShortened.csv",
        "EmoLex": "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "EmoVAD": "NRC-VAD-Lexicon.txt",
        "EmoAff": "NRC-VAD-Lexicon.txt",
        "HSsent": "HS-unigrams.txt",
        "MPQA": "MPQA_sentiment.csv"
    }

    def __init__(self, dir = "wordlists/") -> None:
        self.dir = dir

    def get_path(self, key) -> str:
        return self.dir + self.wordlists[key]

    def load_EmoVAD(self) -> pd.DataFrame:
        return pd.read_csv(self.get_path("EmoVAD"), names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t')





        