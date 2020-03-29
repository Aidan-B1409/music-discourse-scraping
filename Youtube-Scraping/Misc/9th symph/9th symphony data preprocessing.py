# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:01:21 2019

@author: Blaize
"""

#load text
filename = '9thsymph.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

#split int words by whitespace
words = text.split()
print(words[:500])

#convert to lowercase
words = [word.lower() for word in words]

# remove punctuation from each word
import string
table = str.maketrans("", "" , string.punctuation)
stripped = [w.translate(table) for w in words]
print(stripped[:100])

# removing stop words
final = [word for word in final if word.isalpha()]
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
final = [w for w in final if not w in stop_words]
print(final[:200])

final = [word for word in final if word.isalpha()]
from nltk.corpus import stopwords
stop_words = set(stopwords.words('spanish'))
final = [w for w in final if not w in stop_words]

final = [word for word in final if word.isalpha()]
from nltk.corpus import stopwords
stop_words = set(stopwords.words('italian'))
final = [w for w in final if not w in stop_words]

#removing gibberish words
final = []
from nltk.corpus import words
for x in stripped:
    if x in words.words():
        final.append(x)
print(final[:100])



# write final clean list to txt file
myfile = open('9thcleantxt.txt', 'w')

for element in final:
    myfile.write(element)
    myfile.write('\n')
myfile.close()

import pandas as pd
import numpy as np
import nltk
df = pd.DataFrame(np.array(final), columns = ['Words'])

dataset = open("9thcleantxt.txt", "r")
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