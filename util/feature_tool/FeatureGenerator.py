from multidataset_wlist import MultiDataset_wlist
from emoaff_wlist import EmoAff_wlist
from EmoVAD_wlist import EmoVAD_wlist
from bsmvad_wlist import BSMVAD_wlist
from mpqa_wlist import MPQA_wlist
from emolex_wlist import EmoLex_wlist
from meta_generator import MetaGenerator
from glob_maker import make_glob
import sys
import time


class FeatureGenerator:

    def __init__(self, song_df, event, wordlists) -> None:
        self.wordlists = wordlists
        self.song_df = song_df
        self.glob_df = make_glob(song_df)
        self.event = event

    def _build_wordlists(self) -> list:
        wlists = []
        wlists.append(EmoVAD_wlist(self.wordlists['EmoVAD']))
        wlists.append(BSMVAD_wlist(self.wordlists['ANEW_Extended']))
        wlists.append(MPQA_wlist(self.wordlists['MPQA']))
        wlists.append(EmoLex_wlist(self.wordlists['EmoLex']))
        wlists.append(EmoAff_wlist(self.wordlists['EmoAff']))
        wlists.append(MultiDataset_wlist(self.wordlists))
        return wlists

    def get_features(self) -> dict:
        features = {}
        # Note that the metaGenerator also drops all blank rows after extracting static metadata, but BEFORE calculating n_comments, etc.
        # Nasty side effect, but it's the best way to shoehorn that into the pipeline without getting inaccurate n_comment readings. \
        features.update(MetaGenerator(self.song_df, self.glob_df).get_features())
        
        for wlist in self._build_wordlists():
            wordlist_tic = time.perf_counter()
            if self.event.wait(0):
                sys.exit()
            features.update(wlist.wordlevel_analysis(self.song_df, self.glob_df))
            # for i, row in enumerate(self.song_df['Comment Body']):
            #     if self.event.wait(0):
            #         sys.exit()
            #     wlist.process_comment(i, row)
            # features.update(wlist.analyze_comments())
            wordlist_toc = time.perf_counter()
            print(f'wordlist {wlist} generation time: {wordlist_toc - wordlist_tic}\n')
        return features

