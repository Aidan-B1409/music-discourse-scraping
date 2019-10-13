# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:08:10 2019

@author: Blaize
"""
import pandas as pd
import numpy as np

#convert list to data frame
df = pd.DataFrame(np.array(final), columns = ['Words'])




import nltk


dataset = open("cleantxt.txt", "r")
comments = dataset.read()

def tokenize():
    if comments is not None:
        words = comments.lower().split()
        return words
    else:
        return None

def map_comments(tokens):
    hash_map = {}
    
    if tokens is not None:
        for element in tokens:
            word = element.replace(",","")
            word = word.replace(".","")
            
            if word in hash_map:
                hash_map[word] = hash_map[word] + 1
            else:
                hash_map[word] = 1
            
        return hash_map
    else:
        return None
    
words = tokenize()

map = map_comments(words)

for word in words:
    print('Word: [' + word + '] Frequency: ' + str(map[word]))