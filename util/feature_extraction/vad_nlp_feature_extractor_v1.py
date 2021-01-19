# vad_nlp_feature_extractor.py
# Patrick Donnelly
# 03/01/2020
# Usage:   python vad_nlp_feature_extractor.py dir wordlist_dir outfile.csv
# Example: python vad_nlp_feature_extractor.py ./train ./wordlists features.csv

import numpy as np
import pandas as pd
import statistics

from pandas._config import config
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import traceback
import argparse
import sys
import os
import re
import configparser

# global tools
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stop_words = stopwords.words('english')  
nltk.download('punkt')
nltk.download('wordnet')

#global column references. Allows for column references to be customized based on config file
comment_keyname = "invalid"
query_index_keyname = "invalid"

def get_config(config_path: str) -> None:
    config = configparser.ConfigParser()
    config.read(config_path)
    global comment_keyname, query_index_keyname
    comment_keyname = config['COLUMN_HEADERS']['COMMENT']
    query_index_keyname = config['COLUMN_HEADERS']['QUERY_INDEX']

# Count comments and words in unprocessed comment list
def word_count(comment_df):
    global comment_keyname, query_index_keyname
    word_count_lst = [len(str(x).split()) for x in comment_df[comment_keyname].tolist()]
    comment_count = len(word_count_lst)
    comment_len = sum(word_count_lst)
    comment_len_mean = statistics.mean(word_count_lst)
    comment_len_stdev = 0
    if len(word_count_lst) > 1:
        comment_len_stdev = statistics.stdev(word_count_lst)
    
    # format result string
    wc_feats = ''
    wc_feats += '{:f}'.format(comment_count) + ',' 
    wc_feats += '{:f}'.format(comment_len) + ',' 
    wc_feats += '{:f}'.format(comment_len_mean) + ',' 
    wc_feats += '{:f}'.format(comment_len_stdev) + ',' 
    
    #print(wc_feats)    
    return wc_feats  
    
# Remove non-ascii characters and parse comments into word list
def clean_and_split_comments(comment_df):
    global comment_keyname, query_index_keyname
    # extract comments and join as one big string
    comment_df[comment_keyname] = comment_df[comment_keyname].astype(str)
    comment = (comment_df.groupby([query_index_keyname])[comment_keyname].apply(lambda x: ','.join(x)).reset_index())[comment_keyname][0]   
    
    # clean the text
    comment = re.sub('<.*?>', '', comment)    # remove HTML tags
    comment = re.sub(r'[^\w\s\']', ' ', comment) # remove punc./non-English
    comment = re.sub(r'\d+','',comment)       # remove numbers
    comment = comment.lower()                 # lower case
    
    # tokenize, lemmatize, and remove stopwords
    word_list = nltk.word_tokenize(comment)  
    word_list = [lemmatizer.lemmatize(w,'v') for w in word_list]    
    word_list = [word for word in word_list if word not in stop_words]
    
    word_dict = {}
    # count number of times each word occurs and put in dictionary
    for word in word_list: 
        word_dict[word] = word_dict.get(word, 0) + 1
    
    # make a dataframe of the counts
    word_df = pd.DataFrame(list(word_dict.items()), columns=['Word', 'Count'])
    
    return word_df, word_list, word_dict
    
# Load the BSM VAD dataset    
def load_bsmvad_words(filepath_bsm):
    # load bsm VAD words
    bsm_df = pd.read_csv(filepath_bsm, encoding='utf-8', engine='python')
    # drop unneeded columns
    bsm_df = bsm_df.drop(bsm_df.iloc[:, 10:64].columns, axis = 1) 
    bsm_df = bsm_df.drop(['V.Rat.Sum', 'A.Rat.Sum','D.Rat.Sum'], axis = 1) 
    # drop blank rows, if any
    bsm_df = bsm_df[bsm_df['Word'].notnull()]
    # lemmatize words  <-- not sure we want
    #bsm_df['Word'] = bsm_df['Word'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    return bsm_df
    
# Calculate the BSM VAD aggregates
def calculate_bsmvad(word_df,bsmvad_df):
    bsmvad_words_df = pd.merge(word_df,bsmvad_df,on='Word')
    #print(bsmvad_words_df)    
    
    # create columns with our totals by word
    bsmvad_words_df['V.Mean.Sum.Total'] = bsmvad_words_df['Count'] * bsmvad_words_df['V.Mean.Sum']
    bsmvad_words_df['V.SD.Sum.Total'] = bsmvad_words_df['Count'] * bsmvad_words_df['V.SD.Sum']
    bsmvad_words_df['A.Mean.Sum.Total'] = bsmvad_words_df['Count'] * bsmvad_words_df['A.Mean.Sum']
    bsmvad_words_df['A.SD.Sum.Total'] = bsmvad_words_df['Count'] * bsmvad_words_df['A.SD.Sum']
    bsmvad_words_df['D.Mean.Sum.Total'] = bsmvad_words_df['Count'] * bsmvad_words_df['D.Mean.Sum']
    bsmvad_words_df['D.SD.Sum.Total'] = bsmvad_words_df['Count'] * bsmvad_words_df['D.SD.Sum'] 
    
    V_Mean_Sum_Total = V_SD_Sum_Total = 0
    A_Mean_Sum_Total = A_SD_Sum_Total = 0
    D_Mean_Sum_Total = D_SD_Sum_Total = 0
    
    # extract the totals
    count_sum =  bsmvad_words_df['Count'].sum() 
    if count_sum > 0:
        V_Mean_Sum_Total =  bsmvad_words_df['V.Mean.Sum.Total'].sum() / count_sum
        V_SD_Sum_Total =  bsmvad_words_df['V.SD.Sum.Total'].sum() / count_sum

        A_Mean_Sum_Total =  bsmvad_words_df['A.Mean.Sum.Total'].sum() / count_sum
        A_SD_Sum_Total =  bsmvad_words_df['A.SD.Sum.Total'].sum() / count_sum

        D_Mean_Sum_Total =  bsmvad_words_df['D.Mean.Sum.Total'].sum() / count_sum
        D_SD_Sum_Total =  bsmvad_words_df['D.SD.Sum.Total'].sum() / count_sum    
    
    # format result string
    bsmvad_feats = ''
    bsmvad_feats += '{:f}'.format(len(bsmvad_words_df)) + ',' 
    bsmvad_feats += '{:f}'.format(V_Mean_Sum_Total) + ',' 
    bsmvad_feats += '{:f}'.format(V_SD_Sum_Total) + ',' 
    bsmvad_feats += '{:f}'.format(A_Mean_Sum_Total) + ',' 
    bsmvad_feats += '{:f}'.format(A_SD_Sum_Total) + ','
    bsmvad_feats += '{:f}'.format(D_Mean_Sum_Total) + ','
    bsmvad_feats += '{:f}'.format(D_SD_Sum_Total) + ','
    
    #print(bsmvad_feats)    
    return bsmvad_feats
    
# Load the NRC VAD dataset    
def load_emovad_words(filepath_emovad):   
    emovad_df = pd.read_csv(filepath_emovad,  names=['Word','Valence','Arousal','Dominance'], skiprows=1, sep='\t')    
    #print(emovad_df)  
    return emovad_df

# Calculate the NRC VAD aggregates
def calculate_emovad(word_df,emovad_df):
    emovad_words_df = pd.merge(word_df,emovad_df,on='Word')
    #print(emovad_words_df)     
    
    # create columns with our totals by word
    emovad_words_df['V.Total'] = emovad_words_df['Count'] * emovad_words_df['Valence']
    emovad_words_df['A.Total'] = emovad_words_df['Count'] * emovad_words_df['Arousal']
    emovad_words_df['D.Total'] = emovad_words_df['Count'] * emovad_words_df['Dominance']
  
    # extract the totals
    count_sum = emovad_words_df['Count'].sum()    
    V_Total = A_Total = D_Total = 0
    V_Mean = V_Std = 0
    A_Mean = A_Std = 0
    D_Mean = D_Std = 0    
        
    
    if count_sum > 0:
        V_Total = emovad_words_df['V.Total'].sum() / count_sum
        A_Total = emovad_words_df['A.Total'].sum() / count_sum
        D_Total = emovad_words_df['D.Total'].sum() / count_sum
        V_Mean  = emovad_words_df['Valence'].mean()       
        A_Mean  = emovad_words_df['Arousal'].mean()        
        D_Mean  = emovad_words_df['Dominance'].mean()
    if count_sum > 1:
        V_Std   = emovad_words_df['Valence'].std()
        A_Std   = emovad_words_df['Arousal'].std()        
        D_Std   = emovad_words_df['Dominance'].std()
        
    # format result string
    emovad_feats = ''
    emovad_feats += '{:f}'.format(len(emovad_words_df)) + ','     
    emovad_feats += '{:f}'.format(V_Total) + ',' 
    emovad_feats += '{:f}'.format(A_Total) + ',' 
    emovad_feats += '{:f}'.format(D_Total) + ',' 
    emovad_feats += '{:f}'.format(V_Mean) + ','
    emovad_feats += '{:f}'.format(V_Std) + ','
    emovad_feats += '{:f}'.format(A_Mean) + ','
    emovad_feats += '{:f}'.format(A_Std) + ','
    emovad_feats += '{:f}'.format(D_Mean) + ','
    emovad_feats += '{:f}'.format(D_Std) + ','
    
    #print(emovad_feats)
    return emovad_feats
    
# Load the NRC affect dataset    
def load_emoaff_words(filepath_emovad):   
    emoaff_df = pd.read_csv(filepath_emoaff,  names=['Word','Score','Affect'], skiprows=1, sep='\t')    
    #print(emoaff_df)   
    return emoaff_df
    
# Calculate the NRC affect aggregates
def calculate_emoaff(word_df,emoaff_df):
    emoaff_words_df = pd.merge(word_df,emoaff_df,on='Word')
    #print(emoaff_words_df)    
    
    # create columns with our totals by word
    emoaff_words_df['Total'] = emoaff_words_df['Count'] * emoaff_words_df['Score']    

    # declare vars
    anger_mean, anger_mean_wt = 0,0
    fear_mean, fear_mean_wt = 0,0
    sadness_mean, sadness_mean_wt = 0,0
    joy_mean, joy_mean_wt = 0,0
    
    # extract the totals
    count_sum = emoaff_words_df['Count'].sum() 
    anger_df = emoaff_words_df[(emoaff_words_df['Affect'] == 'anger')]    
    if len(anger_df) > 0:
        anger_mean = anger_df['Total'].sum() / anger_df['Count'].sum()
        anger_mean_wt = anger_df['Total'].sum() / count_sum     
    fear_df = emoaff_words_df[(emoaff_words_df['Affect'] == 'fear')]
    if len(fear_df) > 0:
        fear_mean = fear_df['Total'].sum() / fear_df['Count'].sum()    
        fear_mean_wt = fear_df['Total'].sum() / count_sum        
    sadness_df = emoaff_words_df[(emoaff_words_df['Affect'] == 'sadness')]
    if len(sadness_df) > 0:      
        sadness_mean = sadness_df['Total'].sum() / sadness_df['Count'].sum()
        sadness_mean_wt = sadness_df['Total'].sum() / count_sum    
    joy_df = emoaff_words_df[(emoaff_words_df['Affect'] == 'joy')]
    if len(joy_df) > 0:
        joy_mean = joy_df['Total'].sum() / joy_df['Count'].sum()
        joy_mean_wt = joy_df['Total'].sum() / count_sum      
    
    # format result string
    emoaff_feats = ''
    emoaff_feats += '{:f}'.format(len(emoaff_words_df)) + ',' 
    emoaff_feats += '{:f}'.format(anger_mean) + ',' 
    emoaff_feats += '{:f}'.format(fear_mean) + ',' 
    emoaff_feats += '{:f}'.format(sadness_mean) + ',' 
    emoaff_feats += '{:f}'.format(joy_mean) + ','
    emoaff_feats += '{:f}'.format(anger_mean_wt) + ',' 
    emoaff_feats += '{:f}'.format(fear_mean_wt) + ',' 
    emoaff_feats += '{:f}'.format(sadness_mean_wt) + ',' 
    emoaff_feats += '{:f}'.format(joy_mean_wt) + ','  
    #print(emoaff_feats)
    return emoaff_feats
    
    
# Load the NRC lexicon dataset    
def load_emolex_words(filepath_emolex):   
    emolex_df = pd.read_csv(filepath_emolex,  names=['Word','Emotion','Association'], skiprows=1, sep='\t') 
    #print(emolex_df) 
    return emolex_df    

# Calculate the NRC lexicon aggregates
def calculate_emolex(word_df,emolex_df):
    emolex_words_df = pd.merge(word_df,emolex_df,on='Word')
    indexes = emolex_words_df[emolex_words_df['Association'] == 0 ].index
    emolex_words_df.drop(indexes , inplace=True)

    emolex_words_df['Total'] = emolex_words_df['Count'] * emolex_words_df['Association']        
    count_sum = emolex_words_df['Count'].sum()    
    #print(emolex_words_df)   
      
      
    joy_prop = anticipation_prop = trust_prop = 0
    surprise_prop = positive_prop = negative_prop = 0
    sadness_prop = fear_prop = disgust_prop = 0
    
    joy_df = emolex_words_df[(emolex_words_df['Emotion'] == 'joy')]
    if len(joy_df) > 0:    
        joy_prop = joy_df['Total'].sum() / count_sum
    
    trust_df = emolex_words_df[(emolex_words_df['Emotion'] == 'trust')]
    if len(trust_df) > 0:    
        trust_prop = trust_df['Total'].sum() / count_sum 
        
    anticipation_df = emolex_words_df[(emolex_words_df['Emotion'] =='anticipation')]
    if len(anticipation_df) > 0:     
        anticipation_prop = anticipation_df['Total'].sum() / count_sum
    
    surprise_df = emolex_words_df[(emolex_words_df['Emotion'] == 'surprise')]  
    if len(surprise_df) > 0:
        surprise_prop = surprise_df['Total'].sum() / count_sum
        
    positive_df = emolex_words_df[(emolex_words_df['Emotion'] == 'positive')]
    if len(positive_df) > 0:    
        positive_prop = positive_df['Total'].sum() / count_sum
    
    negative_df = emolex_words_df[(emolex_words_df['Emotion'] == 'negative') ]   
    if len(negative_df) > 0:
        negative_prop = negative_df['Total'].sum() / count_sum    
    
    sadness_df = emolex_words_df[(emolex_words_df['Emotion'] == 'sadness')]
    if len(sadness_df) > 0:    
        sadness_prop = sadness_df['Total'].sum() / count_sum  
 
    fear_df = emolex_words_df[(emolex_words_df['Emotion'] == 'fear')]    
    if len(fear_df) > 0:
        fear_prop = fear_df['Total'].sum() / count_sum 
 
    disgust_df = emolex_words_df[(emolex_words_df['Emotion'] == 'disgust')]
    if len(disgust_df) > 0:    
        disgust_prop = disgust_df['Total'].sum() / count_sum
    
    # format result string
    emolex_feats = ''
    emolex_feats += '{:f}'.format(len(emolex_words_df)) + ',' 
    emolex_feats += '{:f}'.format(joy_prop) + ',' 
    emolex_feats += '{:f}'.format(trust_prop) + ',' 
    emolex_feats += '{:f}'.format(anticipation_prop) + ','
    emolex_feats += '{:f}'.format(surprise_prop) + ','
    emolex_feats += '{:f}'.format(positive_prop) + ',' 
    emolex_feats += '{:f}'.format(negative_prop) + ',' 
    emolex_feats += '{:f}'.format(sadness_prop) + ',' 
    emolex_feats += '{:f}'.format(fear_prop) + ',' 
    emolex_feats += '{:f}'.format(disgust_prop) + ','
    #print(emolex_feats)
    return emolex_feats    
    
# Load the HS sentiment dataset    
def load_hssent_words(filepath_HSsent):   
    hssent_df = pd.read_csv(filepath_HSsent,  names=['Word','Rating','pCount','nCount'], skiprows=0, sep='\t') 
    
    hssent_df['Word'] = [re.sub(r'[^\w\s\']', '', str(x)) for x in hssent_df['Word']]
    hssent_df = hssent_df.drop(hssent_df.iloc[:, 2:4].columns, axis = 1) 
    #print(hssent_df)  
    return hssent_df    
    
# Calculate the HSsent  aggregates
def calculate_hssent(word_df,hssent_df):    
    hssent_words_df = pd.merge(word_df,hssent_df,on='Word')
    hssent_words_df['Total'] = hssent_words_df['Count'] * hssent_words_df['Rating']    # extract the totals        
    count_sum = hssent_words_df['Count'].sum()   
    print(count_sum)
    if count_sum > 0:
        hssent_uniq_avg = hssent_words_df['Rating'].mean()
        hssent_uniq_std = hssent_words_df['Rating'].std()
        hssent_dupl_avg = hssent_words_df['Total'].sum() / count_sum
    #print(hssent_words_df)   
    
    words_clean_unique_n = len(word_df)
    words_clean_total_n = word_df['Count'].sum()
    hssent_uniq_word_ratio = len(hssent_words_df) / len(word_df)
    hssent_dupl_word_ratio = count_sum / word_df['Count'].sum()
    
    hssent_feats = ''
    #hssent_feats += '{:f}'.format(len(hssent_words_df)) + ',' 
    #hssent_feats += '{:f}'.format(count_sum) + ','     
    #hssent_feats += '{:f}'.format(hssent_uniq_word_ratio) + ',' 
    #hssent_feats += '{:f}'.format(hssent_dupl_word_ratio) + ',' 
    hssent_feats += '{:f}'.format(len(hssent_words_df)) + ',' 
    hssent_feats += '{:f}'.format(hssent_uniq_avg) + ',' 
    hssent_feats += '{:f}'.format(hssent_uniq_std) + ',' 
    hssent_feats += '{:f}'.format(hssent_dupl_avg) + ',' 
    #print(hssent_feats)
    return hssent_feats
    

# Load the ANEW sentiment dataset    
def load_anew_words(filepath_anew):   
    anew_df = pd.read_csv(filepath_anew,  names=['Word','Valence','Arousal','Dominance'], skiprows=1) 
    return anew_df
    
# Calculate the ANEW aggregates
def calculate_anew(word_df,anew_df):     
    anew_words_df = pd.merge(word_df,anew_df,on='Word')
    anew_words_df['V.Total'] = anew_words_df['Count'] * anew_words_df['Valence'] 
    anew_words_df['A.Total'] = anew_words_df['Count'] * anew_words_df['Arousal'] 
    anew_words_df['D.Total'] = anew_words_df['Count'] * anew_words_df['Dominance'] 
    
    # extract the totals
    count_sum = anew_words_df['Count'].sum()
    
    V_Total = A_Total = D_Total = 0
    V_Mean = V_Std = 0
    A_Mean = A_Std = 0
    D_Mean = D_Std = 0    
    if count_sum > 0:
        V_Total = anew_words_df['V.Total'].sum() / count_sum
        A_Total = anew_words_df['A.Total'].sum() / count_sum
        D_Total = anew_words_df['D.Total'].sum() / count_sum
        
        V_Mean  = anew_words_df['Valence'].mean()        
        A_Mean  = anew_words_df['Arousal'].mean()        
        D_Mean  = anew_words_df['Dominance'].mean()
        
    if count_sum > 1:
        V_Std   = anew_words_df['Valence'].std()
        A_Std   = anew_words_df['Arousal'].std()   
        D_Std   = anew_words_df['Dominance'].std()        
    
    # format result string
    anew_feats = ''
    anew_feats += '{:f}'.format(len(anew_words_df)) + ',' 
    anew_feats += '{:f}'.format(V_Total) + ',' 
    anew_feats += '{:f}'.format(A_Total) + ',' 
    anew_feats += '{:f}'.format(D_Total) + ',' 
    anew_feats += '{:f}'.format(V_Mean) + ','
    anew_feats += '{:f}'.format(V_Std) + ','
    anew_feats += '{:f}'.format(A_Mean) + ','
    anew_feats += '{:f}'.format(A_Std) + ','
    anew_feats += '{:f}'.format(D_Mean) + ','
    anew_feats += '{:f}'.format(D_Std) + ','
    return anew_feats
    
# Load the MPQA sentiment dataset    
def load_mpqa_words(filepath_mpqa):   
    mpqa_df = pd.read_csv(filepath_mpqa,  names=['Word','Sentiment'], skiprows=0)     
    return mpqa_df   
    
# Calculate the MPQA aggregates
def calculate_mpqa(word_df,mpqa_df):     
    mpqa_words_df = pd.merge(word_df,mpqa_df,on='Word')   
    mpqa_words_df.drop_duplicates(inplace=True)
    #print(mpqa_words_df)
    
    count_sum = mpqa_words_df['Count'].sum()    
    positive_prop = neutral_prop = negative_prop = 0
    
    positive_df = mpqa_words_df[(mpqa_words_df['Sentiment'] == 'positive')]   
    if len(positive_df) > 0:    
        positive_prop = positive_df['Count'].sum() / count_sum   
    neutral_df = mpqa_words_df[(mpqa_words_df['Sentiment'] == 'neutral')] 
    if len(neutral_df) > 0:    
        neutral_prop = neutral_df['Count'].sum() / count_sum   
    negative_df = mpqa_words_df[(mpqa_words_df['Sentiment'] == 'negative')] 
    if len(negative_df) > 0:    
        negative_prop = negative_df['Count'].sum() / count_sum           
  
    # format result string
    mpqa_feats = ''
    mpqa_feats += '{:f}'.format(len(mpqa_words_df)) + ',' 
    mpqa_feats += '{:f}'.format(positive_prop) + ',' 
    mpqa_feats += '{:f}'.format(neutral_prop) + ',' 
    mpqa_feats += '{:f}'.format(negative_prop) + ',' 
    return mpqa_feats   
    
# argument parser
parser = argparse.ArgumentParser(description='Load each file in a directory, cleans and combines the comment text, matches words against several VAD word lists, and aggregates VADs of matching words to extract features.')
parser.add_argument('dir', help='comment directory')
parser.add_argument('wordlists', help='word lists directory')
parser.add_argument('outfile', help='output csv file')
parser.add_argument('configfile', help='Directory of headerconfig.ini file')

# default directories
comment_dir = './dat/'
wordlist_dir = './wordlists/'
config_dir = './config/'
filepath_bsmvad = wordlist_dir+'/'+'BRM-emot-submit.csv'
filepath_emolex = wordlist_dir+'/'+'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
filepath_emovad = wordlist_dir+'/'+'NRC-VAD-Lexicon.txt'
filepath_emoaff = wordlist_dir+'/'+'NRC-AffectIntensity-Lexicon.txt'

def main():
    args = parser.parse_args()    
    wordlist_dir = args.wordlists + '/'
    comment_dir = args.dir + '/'
    config_dir = args.configfile 

    #load headers config
    get_config(config_dir)
    
    # file paths
    filepath_bsmvad = wordlist_dir+'/'+'BRM-emot-submit.csv'
    filepath_emolex = wordlist_dir+'/'+'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    filepath_emovad = wordlist_dir+'/'+'NRC-VAD-Lexicon.txt'
    filepath_emoaff = wordlist_dir+'/'+'NRC-AffectIntensity-Lexicon.txt'
    filepath_HSsent = wordlist_dir+'/'+'HS-unigrams.txt'
    filepath_anew = wordlist_dir+'/'+'ANEW_EnglishShortened.csv'
    filepath_mpqa = wordlist_dir+'/'+'MPQA_sentiment.csv'

    # load word lists
    bsmvad_df = load_bsmvad_words(filepath_bsmvad)
    emovad_df = load_emovad_words(filepath_emovad)
    emoaff_df = load_emoaff_words(filepath_emoaff)
    emolex_df = load_emolex_words(filepath_emolex)
    hssent_df = load_hssent_words(filepath_HSsent)
    anew_df = load_anew_words(filepath_anew)
    mpqa_df = load_mpqa_words(filepath_mpqa)
    
    out = open(args.outfile,'w+') 
    # write csv header
    out.write('song_id,')
    out.write('comment_n,comment_len,comment_len_mean,comment_len_stdev,')
    out.write('words_clean_unique_n,words_clean_total_n,')
    out.write('bsm_n_words,')
    out.write('bsm_v_mean,bsm_v_std,bsm_a_mean,bsm_a_std,bsm_d_mean,bsm_d_std,')
    out.write('emo_n_words,')
    out.write('emo_v_total,emo_v_mean,emo_v_std,')
    out.write('emo_a_total,emo_a_mean,emo_a_std,')
    out.write('emo_d_total,emo_d_mean,emo_d_std,')
    out.write('aff_n_words,')
    out.write('aff_anger,aff_fear,aff_sadness,aff_joy,')

    out.write('aff_anger_wt,aff_fear_wt,aff_sadness_wt,aff_joy_wt,')
    out.write('lex_n_words,')
    out.write('lex_joy,lex_trust,lex_anticipation,lex_surprise,lex_positive,lex_negative,lex_sadness,lex_fear,lex_disgust,')
    out.write('hssent_n_words,hssent_uniq_avg,hssent_uniq_std,hssent_dupl_avg,')
    out.write('anew_n_words,')
    out.write('anew_v_total,anew_v_mean,anew_v_std,')
    out.write('anew_a_total,anew_a_mean,anew_a_std,')
    out.write('anew_d_total,anew_d_mean,anew_d_std,')
    out.write('mpqa_n_words,mpqa_pos,mpqa_neu,mpqa_neg,')
    
    
    out.write('song_valence,song_arousal')
    out.write('\n')
    out.close()

    # new error log
    err = open('error_log.txt','w+')
    err.close()

    # process the dataset
    for filename in os.listdir(comment_dir):
        print(filename)
        if not(filename.endswith(".csv")): 
            print(filename.replace('.csv',''), '- not a csv')
            continue
        # skip empty (no comments)    
        if os.stat(comment_dir+filename).st_size == 0:            
            print(filename.replace('.csv',''), '- empty')
            continue            

        try:
            # read file
            comments_df = pd.read_csv(comment_dir+filename,encoding='utf-8',engine='python', index_col=False)
            #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                    #print(comments_df[comment_keyname])            
            song_ids = comments_df[query_index_keyname].unique()
            for song_id in song_ids:
                print(song_id)
                # comments for just the song_id
                comment_df = comments_df[comments_df[query_index_keyname]==song_id]
                song_valence = comment_df[comment_df[query_index_keyname]==song_id].iloc[0]['Valence']
                song_arousal = comment_df[comment_df[query_index_keyname]==song_id].iloc[0]['Arousal']
                
                # split comments into word list

                word_df, word_lst, word_dict = clean_and_split_comments(comment_df)
                #print(word_df)
                words_clean_unique_n = len(word_dict)
                words_clean_total_n = sum(word_dict.values())
                
                #print(word_lst)
                wc_feats = word_count(comment_df)
                            
                # BSM VAD words
                bsmvad_feats = calculate_bsmvad(word_df,bsmvad_df)

                # NRC VAD words
                emovad_feats = calculate_emovad(word_df,emovad_df)
                
                # NRC affect words
                emoaff_feats = calculate_emoaff(word_df,emoaff_df)  

                # NRC affect words
                emolex_feats = calculate_emolex(word_df,emolex_df) 

                # HS sentiment words
                hssent_feats = calculate_hssent(word_df,hssent_df)                
                # ANEW VAD words    
                anew_feats = calculate_anew(word_df,anew_df)        
                
                # MPQA sentiment words    
                mpqa_feats = calculate_mpqa(word_df,mpqa_df)  
                
                # print to file
                out = open(args.outfile,'a+') 
                out.write(str(song_id) + ',')
                out.write(wc_feats)
                out.write(str(words_clean_unique_n)+','+str(words_clean_total_n)+',')
                out.write(bsmvad_feats)
                out.write(emovad_feats)
                out.write(emoaff_feats)
                out.write(emolex_feats)
                out.write(hssent_feats)
                out.write(anew_feats)
                out.write(mpqa_feats)
                out.write('{:f}'.format(song_valence) + ',' + '{:f}'.format(song_arousal) + ',')
                out.write('\n')
                out.close()
        except Exception as e:
            err = open('error_log.txt','a+') 
            err.write(str(filename) + '\n')
            err.write(str(e)+'\n\n')
            err.close()            
            print(filename.replace('.csv',''), '- problem')
            print(traceback.format_exc())

if __name__ == "__main__":
    main()