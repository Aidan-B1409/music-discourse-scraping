import pandas as pd
import re
import nltk
import functools
from os import walk
from datetime import datetime
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

from wordlists import WordLists
from csv_builder import CSVBuilder


class FeatureExtractor:

    # global tools
    lemmatizer = WordNetLemmatizer()
    nltk.download('stopwords')
    stop_words = stopwords.words('english')  
    nltk.download('punkt')
    nltk.download('wordnet')
    
    def __init__(self, wordlists = WordLists(), comment_path = "") -> None:
        self.wordlists = wordlists
        self.comment_path = comment_path

    def song_csv_generator(self):
        for subdir, dirs, files in walk(self.comment_path):
            for file in files:
                fdir = subdir + "/" + file
                yield pd.read_csv(fdir, encoding="utf-8", index_col = False, engine="c")

    def main(self) -> None:
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        csv_name = "deezer_features_" + timestamp + ".csv"
        with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            csv = CSVBuilder(csvfile)
            for df in self.song_csv_generator():
                #TODO - Need to skip empty csvs - but provide some metadata
                csv.writerow(self.generate_features(df))


    # Generate song level features
    def generate_features(self, df) -> list:
        features = []
        #Metadata
        features.append(str(self.song_id(df)))
        #Split the song name, where list[0] = artist name, list[1] = song name
        artist_and_song_name = self.song_name(df)
        features.append(artist_and_song_name[0])
        features.append(artist_and_song_name[1])        
        features.append(self.n_comments(df))

        self.append_emovad_features(features, df)
        return features

    def append_emovad_features(self, features, song_df) -> None:
        emovad = self.wordlists.load_EmoVAD()
        glob_features = self.glob_features(emovad, song_df)
        for feature in glob_features:
            features.append(feature)

    #for song-level feature generation
    #take all words in the song file, return a dict of the count of each word occurance. '
    #TODO - change me if using this pipeline for n-grams
    def glob_comments(self, comment_df):
        # extract comments and join as one big string
        comment_df['Comment Body'] = comment_df['Comment Body'].astype(str)
        # Drop End Of File
        # TODO - Fix comment puller so it doesn't add this damn End Of File string
        # This is what happens when you ask freshmen to do coding. 
        for word in comment_df['Comment Body']:
            re.sub("End Of File", "", word)

        comment_glob = (comment_df.groupby(['Query Index'])['Comment Body'].apply(lambda x: ','.join(x)).reset_index())['Comment Body'][0]   
    
        comment_glob = self.clean_string(comment_glob)
        
        # tokenize and remove stopwords
        word_list = nltk.word_tokenize(comment_glob)  
        word_list = [word for word in word_list if word not in self.stop_words]
        
        word_dict = {}
        # count number of times each word occurs and put in dictionary
        for word in word_list: 
            word_dict[word] = word_dict.get(word, 0) + 1
        
        # make a dataframe of the counts
        word_df = pd.DataFrame(list(word_dict.items()), columns=['Word', 'Count'])
        
        return word_df, word_list, word_dict

    def clean_string(self, comment_glob):
        # clean the text
        comment_glob = re.sub('<.*?>', '', comment_glob)    # remove HTML tags
        comment_glob = re.sub(r'[^\w\s\']', ' ', comment_glob) # remove punc./non-English
        comment_glob = re.sub(r'\d+','',comment_glob)       # remove numbers
        comment_glob = comment_glob.lower()                 # lower case
        return comment_glob

    #The sum of all matching words from a given wordlist. 
    def sum_words(self, semantic_df):
        return semantic_df['Count'].sum()

    # Find the features of the word glob for a specific song 
    def glob_features(self, emovad_df, song_df) -> list:
        features = []
        words_df, words_list, words_dict = self.glob_comments(song_df)
        semantic_word_df = pd.merge(words_df, emovad_df, on='Word')
        #n_words_uniq - only do once!
        if(len(words_df) == 2):
            print(words_df)

        features.append(len(words_df))
        #EmoVAD_glob_sum - TODO fix off by one error
        features.append(len(words_df))
        semantic_word_df['V_Total'] = semantic_word_df['Count'] * emovad_df['Valence']
        semantic_word_df['A_Total'] = semantic_word_df['Count'] * emovad_df['Arousal']
        semantic_word_df['D_Total'] = semantic_word_df['Count'] * emovad_df['Dominance']
        #VAD means including duplicates
        features.append(semantic_word_df['V_Total'].mean())
        features.append(semantic_word_df['V_Total'].std())
        features.append(semantic_word_df['A_Total'].mean())
        features.append(semantic_word_df['A_Total'].std())
        features.append(semantic_word_df['D_Total'].mean())
        features.append(semantic_word_df['D_Total'].std())

        #VAD means on unique words only
        features.append(semantic_word_df['Valence'].mean())
        features.append(semantic_word_df['Valence'].std())
        features.append(semantic_word_df['Arousal'].mean())
        features.append(semantic_word_df['Arousal'].std())
        features.append(semantic_word_df['Dominance'].mean())
        features.append(semantic_word_df['Dominance'].std())

        return features


    #I want this as an int unlike the other features so that I can use it in following calculations. 
    def n_comments(self, df) -> int:
        #TODO - Get each row that has a valid value in comment body


    def song_id(self, df) -> str:
        return str(df.iloc[0]['Song ID'])

    def song_name(self, df) -> list:
        query = str(df.iloc[0]['Query'])
        artist_and_song_name = query.split('" "')
        for i in range(len(artist_and_song_name)):
            artist_and_song_name[i] = re.sub(r"title:", "", artist_and_song_name[i])
            artist_and_song_name[i] = re.sub(r'"', "", artist_and_song_name[i])
        return artist_and_song_name

 
if __name__ == "__main__":
    fe = FeatureExtractor(comment_path="/mnt/g/new_data/subset_deezer_test")
    fe.main()