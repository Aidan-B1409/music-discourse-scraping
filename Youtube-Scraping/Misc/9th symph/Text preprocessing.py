# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:44:36 2019

@author: Blaize
"""

#load text
filename = 'Tearsinheavencorp.txt'
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

#removing gibberish words
final = []
from nltk.corpus import words
for x in stripped:
    if x in words.words():
        final.append(x)
print(final[:100])

# removing stop words
final = [word for word in final if word.isalpha()]
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
final = [w for w in final if not w in stop_words]
print(final[:200])

# write final clean list to txt file
myfile = open('cleantxt.txt', 'w')

for element in final:
    myfile.write(element)
    myfile.write('\n')
myfile.close()
