from multidataset_wlist import MultiDataset_wlist
from emoaff_wlist import EmoAff_wlist
from EmoVAD_wlist import EmoVAD_wlist
from bsmvad_wlist import BSMVAD_wlist
from mpqa_wlist import MPQA_wlist
from emolex_wlist import EmoLex_wlist
import pandas as pd
from meta_generator import MetaGenerator
from glob_maker import make_glob
from os import getcwd

class FeatureGenerator:

    list_paths = {
        "ANEW_Extended": "BRM-emot-submit.csv",
        "ANEW_Ext_Condensed": "ANEW_EnglishShortened.csv",
        "EmoLex": "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "EmoVAD": "NRC-VAD-Lexicon.txt",
        "EmoAff": "NRC-AffectIntensity-Lexicon.txt",
        "HSsent": "HS-unigrams.txt",
        "MPQA": "MPQA_sentiment.csv"
    }

    def __init__(self, song_df) -> None:
        self.wordlists = self._build_wordlists()
        self.song_df = song_df
        self.glob_df = make_glob(song_df)

    def _build_wordlists(self) -> list:
        wlists = []
        wlists.append(EmoVAD_wlist(getcwd() + '/wordlists/' + self.list_paths['EmoVAD']))
        wlists.append(BSMVAD_wlist(getcwd() + '/wordlists/' + self.list_paths['ANEW_Extended']))
        wlists.append(MPQA_wlist(getcwd() + '/wordlists/' + self.list_paths['MPQA']))
        wlists.append(EmoLex_wlist(getcwd() + '/wordlists/' + self.list_paths['EmoLex']))
        wlists.append(EmoAff_wlist(getcwd() + '/wordlists/' + self.list_paths['EmoAff']))
        wlists.append(MultiDataset_wlist(self.list_paths))
        return wlists

    def get_features(self) -> dict:
        features = {}
        # Note that the metaGenerator also drops all blank rows after extracting static metadata, but BEFORE calculating n_comments, etc.
        # Nasty side effect, but it's the best way to shoehorn that into the pipeline without getting inaccurate n_comment readings. 
        features.update(MetaGenerator(self.song_df, self.glob_df).get_features())
        for wlist in self.wordlists:
            features.update(wlist.wordlevel_analysis(self.song_df, self.glob_df))
            for i, row in enumerate(self.song_df['Comment Body']):
                wlist.process_comment(i, row)
            features.update(wlist.analyze_comments())
        return features

