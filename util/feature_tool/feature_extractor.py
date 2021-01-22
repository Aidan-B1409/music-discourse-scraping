import pandas as pd
import re
import nltk
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

    def clean_and_split_comments(self, comment_df):
        # extract comments and join as one big string
        comment_df['Comment Body'] = comment_df['Comment Body'].astype(str)
        comment = (comment_df.groupby(['Query Index'])['Comment Body'].apply(lambda x: ','.join(x)).reset_index())['Comment Body'][0]   
        
        # clean the text
        comment = re.sub('<.*?>', '', comment)    # remove HTML tags
        comment = re.sub(r'[^\w\s\']', ' ', comment) # remove punc./non-English
        comment = re.sub(r'\d+','',comment)       # remove numbers
        comment = comment.lower()                 # lower case
        
        # tokenize, lemmatize, and remove stopwords
        word_list = nltk.word_tokenize(comment)  
        word_list = [self.lemmatizer.lemmatize(w,'v') for w in word_list]    
        word_list = [word for word in word_list if word not in self.stop_words]
        
        word_dict = {}
        # count number of times each word occurs and put in dictionary
        for word in word_list: 
            word_dict[word] = word_dict.get(word, 0) + 1
        
        # make a dataframe of the counts
        word_df = pd.DataFrame(list(word_dict.items()), columns=['Word', 'Count'])
        
        return word_df, word_list, word_dict

    # Generate song level features
    def generate_features(self, df) -> list:
        features = []
        #Metadata
        features.append(str(self.song_id(df)))
        features.append(self.song_name(df))        
        features.append(self.n_comments(df))
        #This is where the fun begins! 
        word_df, word_list, word_dict = self.clean_and_split_comments(df)
        self.append_emovad_features(features, word_df)
        return features

    #I want this as an int unlike the other features so that I can use it in following calculations. 
    def n_comments(self, df) -> int:
        #We subtract one from the length because the last row is just metadata, not comments. 
        return len(df) - 1

    def song_id(self, df) -> str:
        return str(df.iloc[0]['Song ID'])

    def song_name(self, df) -> str:
        query = str(df.iloc[0]['Query'])
        songname = re.sub(r"title:", "", query)
        songname = re.sub(r'"', "", songname)
        return songname

    def append_emovad_features(self, features, word_df) -> None:
        emovad = self.wordlists.load_EmoVAD()
        features.append(str(self.sum_words(word_df, emovad)))

    #The sum of all matching words from a given wordlist. 
    def sum_words(self, word_df, wordlist_df):
        semwords_from_comments = pd.merge(word_df, wordlist_df, on='Word')
        return semwords_from_comments['Count'].sum()


if __name__ == "__main__":
    fe = FeatureExtractor(comment_path="/mnt/g/new_data/subset_deezer_test")
    fe.main()

