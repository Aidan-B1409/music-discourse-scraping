from matplotlib import pyplot
import pandas as pd
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as pyplot
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image
import re
from itertools import compress
from generate_wordlist import Wordlist

wordlist_dir = 'wordlists'

    #return wordlist

def make_word_cloud(words: str)-> None:
    maskArray = npy.array(Image.open('mask.png'))
    cloud = WordCloud(background_color = 'white' , max_words = 200, mask = maskArray, stopwords = set(STOPWORDS))
    cloud.generate(words)
    cloud.to_file('wordCloud.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="The directory in which all data is currently stored")
    args = parser.parse_args()
    rootdir = args.rootdir
    wordlist = Wordlist(wordlist_dir, rootdir)
    words = wordlist.filtered_to_str()
    make_word_cloud(words)
    

if __name__ == "__main__":
    main()