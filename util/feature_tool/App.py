import pandas as pd
import csv
import sys
import threading
from os import walk
from queue import Queue
from datetime import datetime
from tqdm import tqdm
import yappi

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
                yield pd.read_csv(fdir, encoding="utf-8", index_col = False, engine="c")

    def main(self) -> None:
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        data_csv_name = "deezer_features_" + timestamp + ".csv"
        with open(data_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.DictWriter(csvfile, self.m_features.keys())
            csvwriter.writeheader()

            # Spin up threads
            queue = Queue()
            sigkill = threading.Event()

            for song_df in self.song_csv_generator():
                queue.put(song_df)

            pbar = tqdm(total = queue.qsize())            
            try:
                yappi.start()
                # generator_thread = threading.Thread(target=self.buildqueue, args=(queue, ))
                wthreads = [threading.Thread(target=self.thread_func, args=(queue, csvwriter, sigkill, pbar)) for t in range(MAX_THREAD)]
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

    def thread_func(self, queue, csvwriter, sigkill, pbar):
        while not queue.empty():
            song_df = queue.get()
            queue.task_done()
            if song_df.empty:
                continue
            features = FeatureGenerator(song_df, sigkill).get_features()
            if sigkill.wait(0):
                print("killing threads here")
                sys.kill()
            pbar.update()
            csvwriter.writerow(features)


    def buildqueue(self, queue):
        pass


if __name__ == "__main__":
    fe = App(comment_path="/mnt/g/subset_deezer_test/")
    fe.main()
