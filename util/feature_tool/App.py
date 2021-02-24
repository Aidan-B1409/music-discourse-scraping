import pandas as pd
import csv
import sys
import threading
import yappi
import time

from os import walk
from os import getcwd
from queue import Queue
from datetime import datetime
from tqdm import tqdm


import EmoVAD_wlist
import bsmvad_wlist
import mpqa_wlist
import emolex_wlist
import emoaff_wlist
import multidataset_wlist
from FeatureGenerator import FeatureGenerator
from analysis import analyze_features


# Some terminology to help out whoever has to interpret this 
# song_df -- The raw data, grabbed from the director
# wordlist_df -- The dataframe holding the semantic wordlist
# word_df -- A wordlist which holds each unique comment in the glob/comment and its # of occurances 
# glob -- The concatanation 
MAX_THREAD = 16

class App:

    list_paths = {
        "ANEW_Extended": "BRM-emot-submit.csv",
        "ANEW_Ext_Condensed": "ANEW_EnglishShortened.csv",
        "EmoLex": "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "EmoVAD": "NRC-VAD-Lexicon.txt",
        "EmoAff": "NRC-AffectIntensity-Lexicon.txt",
        "HSsent": "HS-unigrams.txt",
        "MPQA": "MPQA_sentiment.csv"
    }

    m_features = {'Song_ID': "", 'Song_Name': "", 'n_words': -1, 'comment_length_stdev': -1, 'Song_Artist': "",
     'existing_valence': "", 'existing_arousal': "", 'n_words_uniq': -1, 'n_comments': -1, 'comment_length_mean': -1}
    wordlists = {EmoVAD_wlist, bsmvad_wlist, mpqa_wlist, emolex_wlist, emoaff_wlist, multidataset_wlist}
    for wlist in wordlists:
        m_features.update(wlist.get_header())
    
    def __init__(self,comment_path = "") -> None:
        self.comment_path = comment_path

    def song_csv_generator(self):
        for subdir, dirs, files in walk(self.comment_path):
            for file in files:
                fdir = subdir + "/" + file
                yield fdir

    def main(self) -> None:
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        data_csv_name = "deezer_features_" + timestamp + ".csv"
        with open(data_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.DictWriter(csvfile, self.m_features.keys())
            csvwriter.writeheader()

            queue = Queue()
            sigkill = threading.Event()

            for song_fname in self.song_csv_generator():
                queue.put(song_fname)

            pbar = tqdm(total = queue.qsize())      

            wordlists = {
                'EmoVAD': pd.read_csv(self._get_wlist_path(self.list_paths['EmoVAD']),
                    names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t'),
                'EmoLex': pd.read_csv(self._get_wlist_path(self.list_paths['EmoLex']),
                    names=['Word','Emotion','Association'], skiprows=1, sep='\t'), 
                'EmoAff': pd.read_csv(self._get_wlist_path(self.list_paths['EmoAff']),
                    names=['Word','Score','Affect'], skiprows=1, sep='\t', index_col=False), 
                'ANEW_Extended': self._load_bsmvad(self._get_wlist_path(self.list_paths['ANEW_Extended'])),
                'MPQA': pd.read_csv(self._get_wlist_path(self.list_paths['MPQA']),
                    names=['Word','Sentiment'], skiprows=0)
            }

            # Spin up threads      
            try:
                yappi.start()
                # generator_thread = threading.Thread(target=self.buildqueue, args=(queue, ))
                wthreads = [threading.Thread(target=self.thread_func,
                 args=(queue, csvwriter, sigkill, pbar, wordlists)) for t in range(MAX_THREAD)]
                # generator_thread.start()
                [thread.start() for thread in wthreads]
                # generator_thread.join()
                [thread.join() for thread in wthreads] 

                pbar.close()  
                yappi.stop()

                 # after we finish generating all our features - do some simple analysis
                analysis_csv_name = "feature_analysis" + timestamp + ".csv"
                analyze_features(data_csv_name, analysis_csv_name, self.m_features)  

                with open('out.txt', 'w') as f:
                    ythreads = yappi.get_thread_stats()
                    for thread in ythreads:
                        f.write("Function stats for (%s) (%d): \n" % (thread.name, thread.id))
                        yappi.get_func_stats(ctx_id=thread.id).print_all(out = f)

            except KeyboardInterrupt as e:
                print(f"Interrupt {e} Recieved - Killing Threads")
                sigkill.set()
                yappi.stop()
                pbar.close()

    def thread_func(self, queue, csvwriter, sigkill, pbar, wordlists):
        while not queue.empty():

            song_fname = queue.get()
            queue.task_done()

            song_df = pd.read_csv(song_fname, encoding="utf-8", index_col = False, engine="c")

            if song_df.empty:
                continue

            features = FeatureGenerator(song_df, sigkill, wordlists).get_features()
            if sigkill.wait(0):
                sys.exit()
            pbar.update()
            csvwriter.writerow(features)

    def _get_wlist_path(self, key):
        return getcwd() + '/wordlists/' + key

    def _load_bsmvad(self, path) -> pd.DataFrame:
        bsmvad_df = pd.read_csv(path, encoding='utf-8', engine='python')
        # drop unneeded columns
        bsmvad_df.drop(bsmvad_df.iloc[:, 10:64].columns, axis = 1, inplace = True) 
        bsmvad_df.drop(['V.Rat.Sum', 'A.Rat.Sum','D.Rat.Sum'], axis = 1, inplace = True) 
        # drop blank rows, if any
        bsmvad_df = bsmvad_df[bsmvad_df['Word'].notnull()]
        return bsmvad_df



if __name__ == "__main__":
    fe = App(comment_path="/mnt/g/bigfiles_subset/")
    fe.main()
