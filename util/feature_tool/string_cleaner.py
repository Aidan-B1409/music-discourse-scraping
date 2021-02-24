import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stop_words = stopwords.words('english')  
nltk.download('punkt')
nltk.download('wordnet')

# Flag to enable/disable lemmatization
# Eventually make this a CLI option. 
LEMMATIZE = False

# Accepts raw megastring (both for glob and for raw use with individual comments)
# Returns list of tokenized words
def clean_comment(comment) -> list:
    # clean the text
    comment = re.sub('<.*?>', '', comment)    # remove HTML tags
    comment = re.sub(r'[^\w\s\']', ' ', comment) # remove punc./non-English
    comment = re.sub(r'\d+','',comment)       # remove numbers
    comment = comment.lower()                 # lower case
    # tokenize and remove stopwords
    word_list = nltk.word_tokenize(comment)
    if(LEMMATIZE):
        word_list = [lemmatizer.lemmatize(w,'v') for w in word_list]    
    word_list = [word for word in word_list if word not in stop_words]
    return word_list

# Given a list of words, count the occurences of a word and create a dataframe
#    [Word]    [Count]
# Important to note terminology - word_df is a list of words and counts
# song_df is the raw data
# TODO - This will not work when analyzing n-grams! Find another way!
# Find union of n-gram wordlist and comment strings? 
def make_word_df(comment_list) -> pd.DataFrame:
    word_dict = {}
    # count number of times each word occurs and put in dictionary
    for word in comment_list: 
        word_dict[word] = word_dict.get(word, 0) + 1
    
    # make a dataframe of the counts
    word_df = pd.DataFrame(list(word_dict.items()), columns=['Word', 'Count'])
    return word_df