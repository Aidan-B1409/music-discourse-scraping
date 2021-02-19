import pandas as pd
import csv
import os
from os import walk
from datetime import datetime

from EmoVAD_wlist import EmoVAD_wlist
from FeatureGenerator import FeatureGenerator
from analysis import analyze_features


# Some terminology to help out whoever has to interpret this 
# song_df -- The raw data, grabbed from the director
# wordlist_df -- The dataframe holding the semantic wordlist
# word_df -- A wordlist which holds each unique comment in the glob/comment and its # of occurances 
# glob -- The concatanation 
class App:

    # NOTE - If you're getting "dict contains fields not in fieldnames" errors, it is probably because 
    # you added features/keys within one of the wlist classes without also adding it here
    # I know it's a gross solution but this is the only way to generate the header and lazy load the features. 
    # In the future - please replace this key system with a JSON. 
    m_features = {
        "Song_ID": "",
        "Song_Artist": "", "Song_Name": "",
        "existing_valence": "", "existing_arousal": "",
        "n_comments": -1, "n_words": -1, "n_words_uniq": -1,
        "comment_length_mean": 0, "comment_length_stdev": 0,
        
        "EmoVAD_glob_v_mean": 0.0, "EmoVAD_glob_v_stdev": 0.0,
        "EmoVAD_glob_a_mean": 0.0, "EmoVAD_glob_a_stdev": 0.0,
        "EmoVAD_glob_d_mean": 0.0, "EmoVAD_glob_d_stdev": 0.0,
        "EmoVAD_glob_v_mean_uniq": 0.0, "EmoVAD_glob_v_stdev_uniq": 0.0,
        "EmoVAD_glob_a_mean_uniq": 0.0, "EmoVAD_glob_a_stdev_uniq": 0.0,
        "EmoVAD_glob_d_mean_uniq": 0.0, "EmoVAD_glob_d_stdev_uniq": 0.0,

        "EmoVAD_glob_v_sqmean": 0.0, "EmoVAD_glob_a_sqmean": 0.0, 
        "EmoVAD_glob_d_sqmean": 0.0, "EmoVAD_glob_v_sqmean_uniq": 0.0, 
        "EmoVAD_glob_a_sqmean_uniq": 0.0, "EmoVAD_glob_d_sqmean_uniq": 0.0, 

        "EmoVAD_glob_max_word_v": 0.0, "EmoVAD_glob_max_word_a": 0.0,
        "EmoVAD_glob_max_word_d": 0.0, 
        
        "EmoVAD_glob_min_word_v": 0.0,
        "EmoVAD_glob_min_word_a": 0.0, "EmoVAD_glob_min_word_d": 0.0,

        "EmoVAD_glob_most_word_v": 0.0, "EmoVAD_glob_most_word_a": 0.0,
        "EmoVAD_glob_most_word_d": 0.0, "EmoVAD_glob_most_word_count": 0.0,

        "EmoVAD_glob_v_ratio": 0.0, "EmoVAD_glob_a_ratio": 0.0, 
        "EmoVAD_glob_d_ratio": 0.0, 
        "EmoVAD_glob_v_uniq_ratio": 0.0, "EmoVAD_glob_a_uniq_ratio": 0.0, 
        "EmoVAD_glob_d_uniq_ratio": 0.0, 

        "EmoVAD_v_mean_mean": 0.0, "EmoVAD_v_mean_stdev": 0.0,
        "EmoVAD_v_std_mean": 0.0, "EmoVAD_v_stdev_stdev": 0.0,
        "EmoVAD_v_uniq_mean_mean": 0.0, "EmoVAD_v_uniq_mean_stdev": 0.0,
        "EmoVAD_v_uniq_std_mean": 0.0, "EmoVAD_v_uniq_stdev_stdev": 0.0,

        "EmoVAD_a_mean_mean": 0.0, "EmoVAD_a_mean_stdev": 0.0,
        "EmoVAD_a_std_mean": 0.0, "EmoVAD_a_stdev_stdev": 0.0,
        "EmoVAD_a_uniq_mean_mean": 0.0, "EmoVAD_a_uniq_mean_stdev": 0.0,
        "EmoVAD_a_uniq_std_mean": 0.0, "EmoVAD_a_uniq_stdev_stdev": 0.0,

        "EmoVAD_d_mean_mean": 0.0, "EmoVAD_d_mean_stdev": 0.0,
        "EmoVAD_d_std_mean": 0.0, "EmoVAD_d_stdev_stdev": 0.0,
        "EmoVAD_d_uniq_mean_mean": 0.0, "EmoVAD_d_uniq_mean_stdev": 0.0,
        "EmoVAD_d_uniq_std_mean": 0.0, "EmoVAD_d_uniq_stdev_stdev": 0.0,

        "EmoVAD_max_word_v_mean": 0.0, "EmoVAD_max_word_v_std": 0.0,
        "EmoVAD_max_word_a_mean": 0.0, "EmoVAD_max_word_a_std": 0.0,
        "EmoVAD_max_word_d_mean": 0.0, "EmoVAD_max_word_d_std": 0.0,

        "EmoVAD_max_word_v_sqmean": 0.0, "EmoVAD_max_word_v_sqstd": 0.0,
        "EmoVAD_max_word_a_sqmean": 0.0, "EmoVAD_max_word_a_sqstd": 0.0,
        "EmoVAD_max_word_d_sqmean": 0.0, "EmoVAD_max_word_d_sqstd": 0.0,
        
        "EmoVAD_min_word_v_mean": 0.0, "EmoVAD_min_word_v_std": 0.0,
        "EmoVAD_min_word_a_mean": 0.0, "EmoVAD_min_word_a_std": 0.0,
        "EmoVAD_min_word_d_mean": 0.0, "EmoVAD_min_word_d_std": 0.0,

        "EmoVAD_most_word_v_mean": 0.0, "EmoVAD_most_word_v_std": 0.0,
        "EmoVAD_most_word_a_mean": 0.0, "EmoVAD_most_word_a_std": 0.0,
        "EmoVAD_most_word_d_mean": 0.0, "EmoVAD_most_word_d_std": 0.0,
        "EmoVAD_most_word_count_mean": 0.0, "EmoVAD_most_word_count_std": 0.0,


        "bsmVAD_glob_v_mean": 0.0, "bsmVAD_glob_v_stdev": 0.0,
        "bsmVAD_glob_a_mean": 0.0, "bsmVAD_glob_a_stdev": 0.0,
        "bsmVAD_glob_d_mean": 0.0, "bsmVAD_glob_d_stdev": 0.0,
        "bsmVAD_glob_v_mean_uniq": 0.0, "bsmVAD_glob_v_stdev_uniq": 0.0,
        "bsmVAD_glob_a_mean_uniq": 0.0, "bsmVAD_glob_a_stdev_uniq": 0.0,
        "bsmVAD_glob_d_mean_uniq": 0.0, "bsmVAD_glob_d_stdev_uniq": 0.0,

        "bsmVAD_glob_v_sqmean": 0.0, "bsmVAD_glob_a_sqmean": 0.0, 
        "bsmVAD_glob_d_sqmean": 0.0, "bsmVAD_glob_v_sqmean_uniq": 0.0, 
        "bsmVAD_glob_a_sqmean_uniq": 0.0, "bsmVAD_glob_d_sqmean_uniq": 0.0, 

        "bsmVAD_glob_max_word_v": 0.0, "bsmVAD_glob_max_word_a": 0.0,
        "bsmVAD_glob_max_word_d": 0.0, 
        
        "bsmVAD_glob_min_word_v": 0.0,
        "bsmVAD_glob_min_word_a": 0.0, "bsmVAD_glob_min_word_d": 0.0,

        "bsmVAD_glob_most_word_v": 0.0, "bsmVAD_glob_most_word_a": 0.0,
        "bsmVAD_glob_most_word_d": 0.0, "bsmVAD_glob_most_word_count": 0.0,

        "bsmVAD_glob_v_ratio": 0.0, "bsmVAD_glob_a_ratio": 0.0, 
        "bsmVAD_glob_d_ratio": 0.0, 
        "bsmVAD_glob_v_uniq_ratio": 0.0, "bsmVAD_glob_a_uniq_ratio": 0.0, 
        "bsmVAD_glob_d_uniq_ratio": 0.0, 

        "bsmVAD_v_mean_mean": 0.0, "bsmVAD_v_mean_stdev": 0.0,
        "bsmVAD_v_std_mean": 0.0, "bsmVAD_v_stdev_stdev": 0.0,
        "bsmVAD_v_uniq_mean_mean": 0.0, "bsmVAD_v_uniq_mean_stdev": 0.0,
        "bsmVAD_v_uniq_std_mean": 0.0, "bsmVAD_v_uniq_stdev_stdev": 0.0,

        "bsmVAD_a_mean_mean": 0.0, "bsmVAD_a_mean_stdev": 0.0,
        "bsmVAD_a_std_mean": 0.0, "bsmVAD_a_stdev_stdev": 0.0,
        "bsmVAD_a_uniq_mean_mean": 0.0, "bsmVAD_a_uniq_mean_stdev": 0.0,
        "bsmVAD_a_uniq_std_mean": 0.0, "bsmVAD_a_uniq_stdev_stdev": 0.0,

        "bsmVAD_d_mean_mean": 0.0, "bsmVAD_d_mean_stdev": 0.0,
        "bsmVAD_d_std_mean": 0.0, "bsmVAD_d_stdev_stdev": 0.0,
        "bsmVAD_d_uniq_mean_mean": 0.0, "bsmVAD_d_uniq_mean_stdev": 0.0,
        "bsmVAD_d_uniq_std_mean": 0.0, "bsmVAD_d_uniq_stdev_stdev": 0.0,

        "bsmVAD_max_word_v_mean": 0.0, "bsmVAD_max_word_v_std": 0.0,
        "bsmVAD_max_word_a_mean": 0.0, "bsmVAD_max_word_a_std": 0.0,
        "bsmVAD_max_word_d_mean": 0.0, "bsmVAD_max_word_d_std": 0.0,

        "bsmVAD_max_word_v_sqmean": 0.0, "bsmVAD_max_word_v_sqstd": 0.0,
        "bsmVAD_max_word_a_sqmean": 0.0, "bsmVAD_max_word_a_sqstd": 0.0,
        "bsmVAD_max_word_d_sqmean": 0.0, "bsmVAD_max_word_d_sqstd": 0.0,
        
        "bsmVAD_min_word_v_mean": 0.0, "bsmVAD_min_word_v_std": 0.0,
        "bsmVAD_min_word_a_mean": 0.0, "bsmVAD_min_word_a_std": 0.0,
        "bsmVAD_min_word_d_mean": 0.0, "bsmVAD_min_word_d_std": 0.0,

        "bsmVAD_most_word_v_mean": 0.0, "bsmVAD_most_word_v_std": 0.0,
        "bsmVAD_most_word_a_mean": 0.0, "bsmVAD_most_word_a_std": 0.0,
        "bsmVAD_most_word_d_mean": 0.0, "bsmVAD_most_word_d_std": 0.0,
        "bsmVAD_most_word_count_mean": 0.0, "bsmVAD_most_word_count_std": 0.0,

        'mpqa_glob_positive_mean': 0.0, 'mpqa_glob_negative_mean': 0.0,
        'mpqa_glob_positive_mean_uniq': 0.0, 'mpqa_glob_negative_mean_uniq': 0.0,

        'mpqa_positive_mostfreqword_mean': 0.0, 'mpqa_negative_mostfreqword_mean': 0.0,
        'mpqa_glob_ratio': 0.0, 'mpqa_glob_ratio_uniq': 0.0,

        'mpqa_positive_means_mean': 0.0, 'mpqa_positive_means_std': 0.0,
        'mpqa_positive_means_uniq_mean': 0.0, 'mpqa_positive_means_uniq_std': 0.0,
        'mpqa_negative_means_mean': 0.0, 'mpqa_negative_means_std': 0.0,
        'mpqa_negative_means_uniq_mean': 0.0, 'mpqa_negative_means_uniq_std': 0.0,

        'mpqa_mostfreq_positiveword_means_mean': 0.0, 'mpqa_mostfreq_positiveword_means_std': 0.0,
        'mpqa_mostfreq_negativeword_means_mean': 0.0, 'mpqa_mostfreq_negativeword_means_std': 0.0,
        'mpqa_ratio_mean': 0.0, 'mpqa_ratio_std': 0.0,
        'mpqa_ratio_uniq_mean': 0.0, 'mpqa_ratio_uniq_std': 0.0, 

        'EmoLex_glob_positive_mean': 0.0, 'EmoLex_glob_positive_mean_uniq': 0.0,
        'EmoLex_glob_negative_mean': 0.0, 'EmoLex_glob_negative_mean_uniq': 0.0,
        'EmoLex_glob_anger_mean': 0.0, 'EmoLex_glob_anger_mean_uniq': 0.0,
        'EmoLex_glob_anticipation_mean': 0.0, 'EmoLex_glob_anticipation_mean_uniq': 0.0,
        'EmoLex_glob_disgust_mean': 0.0, 'EmoLex_glob_disgust_mean_uniq': 0.0,
        'EmoLex_glob_fear_mean': 0.0, 'EmoLex_glob_fear_mean_uniq': 0.0,
        'EmoLex_glob_joy_mean': 0.0, 'EmoLex_glob_joy_mean_uniq': 0.0,
        'EmoLex_glob_sadness_mean': 0.0, 'EmoLex_glob_sadness_mean_uniq': 0.0,
        'EmoLex_glob_surprise_mean': 0.0, 'EmoLex_glob_surprise_mean_uniq': 0.0,
        'EmoLex_glob_trust_mean': 0.0, 'EmoLex_glob_trust_mean_uniq': 0.0,

        'EmoLex_glob_most_freq_positive_word_mean': 0.0,
        'EmoLex_glob_most_freq_negative_word_mean': 0.0,
        'EmoLex_glob_most_freq_anger_word_mean': 0.0,
        'EmoLex_glob_most_freq_anticipation_word_mean': 0.0,
        'EmoLex_glob_most_freq_disgust_word_mean': 0.0,
        'EmoLex_glob_most_freq_fear_word_mean': 0.0,
        'EmoLex_glob_most_freq_joy_word_mean': 0.0,
        'EmoLex_glob_most_freq_sadness_word_mean': 0.0,
        'EmoLex_glob_most_freq_surprise_word_mean': 0.0,
        'EmoLex_glob_most_freq_trust_word_mean': 0.0,

        'EmoLex_glob_posneg_ratio': 0.0, 'EmoLex_glob_posneg_ratio_uniq': 0.0, 

        'EmoLex_positive_means_mean': 0.0, 'EmoLex_positive_means_std': 0.0,
        'EmoLex_positive_means_uniq_mean': 0.0, 'EmoLex_positive_means_uniq_std': 0.0,
        'EmoLex_negative_means_mean': 0.0, 'EmoLex_negative_means_std': 0.0,
        'EmoLex_negative_means_uniq_mean': 0.0, 'EmoLex_negative_means_uniq_std': 0.0,
        'EmoLex_anger_means_mean': 0.0, 'EmoLex_anger_means_std': 0.0, 
        'EmoLex_anger_means_uniq_mean': 0.0, 'EmoLex_anger_means_uniq_std': 0.0,
        'EmoLex_anticipation_means_mean': 0.0, 'EmoLex_anticipation_means_std': 0.0,
        'EmoLex_anticipation_means_uniq_mean': 0.0, 'EmoLex_anticipation_means_uniq_std': 0.0,
        'EmoLex_disgust_means_mean': 0.0, 'EmoLex_disgust_means_std': 0.0,
        'EmoLex_disgust_means_uniq_mean': 0.0, 'EmoLex_disgust_means_uniq_std': 0.0,
        'EmoLex_fear_means_mean': 0.0, 'EmoLex_fear_means_std': 0.0,
        'EmoLex_fear_means_uniq_mean': 0.0, 'EmoLex_fear_means_uniq_std': 0.0,
        'EmoLex_joy_means_mean': 0.0, 'EmoLex_joy_means_std': 0.0, 
        'EmoLex_joy_means_uniq_mean': 0.0, 'EmoLex_joy_means_uniq_std': 0.0,
        'EmoLex_sadness_means_mean': 0.0, 'EmoLex_sadness_means_std': 0.0,
        'EmoLex_sadness_means_uniq_mean': 0.0, 'EmoLex_sadness_means_uniq_std': 0.0,
        'EmoLex_surprise_means_mean': 0.0, 'EmoLex_surprise_means_std': 0.0, 
        'EmoLex_surprise_means_uniq_mean': 0.0, 'EmoLex_surprise_means_uniq_std': 0.0,
        'EmoLex_trust_means_mean': 0.0, 'EmoLex_trust_means_std': 0.0, 
        'EmoLex_trust_means_uniq_mean': 0.0, 'EmoLex_trust_means_uniq_std': 0.0,

        'EmoLex_most_freq_positive_word_means_mean': 0.0, 'EmoLex_most_freq_positive_word_means_std': 0.0,
        'EmoLex_most_freq_negative_word_means_mean': 0.0, 'EmoLex_most_freq_negative_word_means_std': 0.0,
        'EmoLex_most_freq_anger_word_means_mean': 0.0, 'EmoLex_most_freq_anger_word_means_std': 0.0,
        'EmoLex_most_freq_anticipation_word_means_mean': 0.0, 'EmoLex_most_freq_anticipation_word_means_std': 0.0,
        'EmoLex_most_freq_disgust_word_means_mean': 0.0, 'EmoLex_most_freq_disgust_word_means_std': 0.0,
        'EmoLex_most_freq_fear_word_means_mean': 0.0, 'EmoLex_most_freq_fear_word_means_std': 0.0,
        'EmoLex_most_freq_joy_word_means_mean': 0.0, 'EmoLex_most_freq_joy_word_means_std': 0.0,
        'EmoLex_most_freq_sadness_word_means_mean': 0.0, 'EmoLex_most_freq_sadness_word_means_std': 0.0,
        'EmoLex_most_freq_surprise_word_means_mean': 0.0, 'EmoLex_most_freq_surprise_word_means_std': 0.0,
        'EmoLex_most_freq_trust_word_means_mean': 0.0, 'EmoLex_most_freq_trust_word_means_std': 0.0,

        'EmoLex_posneg_ratio_mean': 0.0, 'EmoLex_posneg_ratio_std': 0.0, 
        'EmoLex_posneg_ratio_uniq_mean': 0.0, 'EmoLex_posneg_ratio_uniq_std': 0.0,

        'EmoAff_glob_anger_mean': 0.0, 'EmoAff_glob_anger_std': 0.0,
        'EmoAff_glob_fear_mean': 0.0, 'EmoAff_glob_fear_std': 0.0,
        'EmoAff_glob_sadness_mean': 0.0, 'EmoAff_glob_sadness_std': 0.0,
        'EmoAff_glob_joy_mean': 0.0, 'EmoAff_glob_joy_std': 0.0,
        'EmoAff_glob_anger_mean_uniq': 0.0, 'EmoAff_glob_anger_std_uniq': 0.0,
        'EmoAff_glob_fear_mean_uniq': 0.0, 'EmoAff_glob_fear_std_uniq': 0.0,
        'EmoAff_glob_sadness_mean_uniq': 0.0, 'EmoAff_glob_sadness_std_uniq': 0.0,
        'EmoAff_glob_joy_mean_uniq': 0.0, 'EmoAff_glob_joy_std_uniq': 0.0,

        'EmoAff_glob_max_word_anger': 0.0, 'EmoAff_glob_max_word_fear': 0.0, 
        'EmoAff_glob_max_word_sadness': 0.0, 'EmoAff_glob_max_word_joy': 0.0,

        'EmoAff_glob_min_word_anger': 0.0, 'EmoAff_glob_min_word_fear': 0.0, 
        'EmoAff_glob_min_word_sadness': 0.0, 'EmoAff_glob_min_word_joy': 0.0,

        'EmoAff_glob_most_word_anger': 0.0, 'EmoAff_glob_most_word_fear': 0.0, 
        'EmoAff_glob_most_word_sadness': 0.0, 'EmoAff_glob_most_word_joy': 0.0,

        'EmoAff_anger_means_mean': 0.0, 'EmoAff_anger_means_std': 0.0,
        'EmoAff_anger_stds_mean': 0.0, 'EmoAff_anger_stds_std': 0.0, 
        'EmoAff_anger_uniq_means_mean': 0.0, 'EmoAff_anger_uniq_means_std': 0.0,
        'EmoAff_anger_uniq_stds_mean': 0.0, 'EmoAff_anger_uniq_stds_std': 0.0,
        
        'EmoAff_fear_means_mean': 0.0, 'EmoAff_fear_means_std': 0.0,
        'EmoAff_fear_stds_mean': 0.0, 'EmoAff_fear_stds_std': 0.0, 
        'EmoAff_fear_uniq_means_mean': 0.0, 'EmoAff_fear_uniq_means_std': 0.0,
        'EmoAff_fear_uniq_stds_mean': 0.0, 'EmoAff_fear_uniq_stds_std': 0.0,

        'EmoAff_sadness_means_mean': 0.0, 'EmoAff_sadness_means_std': 0.0,
        'EmoAff_sadness_stds_mean': 0.0, 'EmoAff_sadness_stds_std': 0.0, 
        'EmoAff_sadness_uniq_means_mean': 0.0, 'EmoAff_sadness_uniq_means_std': 0.0,
        'EmoAff_sadness_uniq_stds_mean': 0.0, 'EmoAff_sadness_uniq_stds_std': 0.0,

        'EmoAff_joy_means_mean': 0.0, 'EmoAff_joy_means_std': 0.0,
        'EmoAff_joy_stds_mean': 0.0, 'EmoAff_joy_stds_std': 0.0, 
        'EmoAff_joy_uniq_means_mean': 0.0, 'EmoAff_joy_uniq_means_std': 0.0,
        'EmoAff_joy_uniq_stds_mean': 0.0, 'EmoAff_joy_uniq_stds_std': 0.0,

        'EmoAff_anger_max_word_mean': 0.0, 'EmoAff_anger_max_word_std': 0.0, 
        'EmoAff_fear_max_word_mean': 0.0, 'EmoAff_fear_max_word_std': 0.0, 
        'EmoAff_sadness_max_word_mean': 0.0, 'EmoAff_sadness_max_word_std': 0.0, 
        'EmoAff_joy_max_word_mean': 0.0, 'EmoAff_joy_max_word_std': 0.0, 

        'EmoAff_anger_min_word_mean': 0.0, 'EmoAff_anger_min_word_std': 0.0, 
        'EmoAff_fear_min_word_mean': 0.0, 'EmoAff_fear_min_word_std': 0.0, 
        'EmoAff_sadness_min_word_mean': 0.0, 'EmoAff_sadness_min_word_std': 0.0, 
        'EmoAff_joy_min_word_mean': 0.0, 'EmoAff_joy_min_word_std': 0.0, 

        'EmoAff_anger_most_word_mean': 0.0, 'EmoAff_anger_most_word_std': 0.0, 
        'EmoAff_fear_most_word_mean': 0.0, 'EmoAff_fear_most_word_std': 0.0, 
        'EmoAff_sadness_most_word_mean': 0.0, 'EmoAff_sadness_most_word_std': 0.0, 
        'EmoAff_joy_most_word_mean': 0.0, 'EmoAff_joy_most_word_std': 0.0, 
    }
    
    def __init__(self,comment_path = "") -> None:
        self.comment_path = comment_path

    def song_csv_generator(self):
        for subdir, dirs, files in walk(self.comment_path):
            for file in files:
                fdir = subdir + "/" + file
                yield pd.read_csv(fdir, encoding="utf-8", index_col = False, engine="c")

    def main(self) -> None:
        self._find_intersects()
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        data_csv_name = "deezer_features_" + timestamp + ".csv"
        with open(data_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.DictWriter(csvfile, self.m_features.keys())
            csvwriter.writeheader()
            for song_df in self.song_csv_generator():
                features = FeatureGenerator(song_df).get_features()
                csvwriter.writerow(features)

        # after we finish generating all our features - do some simple analysis
        analysis_csv_name = "feature_analysis" + timestamp + ".csv"
        analyze_features(data_csv_name, analysis_csv_name, self.m_features)


    # GARBAGE UTILITY METHOD FOR TESTING INTERSECTIONS BETWEEN WORDLISTS - DELETEME
    def _find_intersects(self):
        bsmvad_df = pd.read_csv(os.getcwd() + '/wordlists/BRM-emot-submit.csv', encoding='utf-8', engine='python')
        # drop unneeded columns
        bsmvad_df.drop(bsmvad_df.iloc[:, 10:64].columns, axis = 1, inplace = True) 
        bsmvad_df.drop(['V.Rat.Sum', 'A.Rat.Sum','D.Rat.Sum'], axis = 1, inplace = True) 
        # drop blank rows, if any
        bsmvad_df = bsmvad_df[bsmvad_df['Word'].notnull()]

        #open emovad
        emovad_df = pd.read_csv(os.getcwd() + '/wordlists/NRC-VAD-Lexicon.txt', names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t')

        print("EmoVAD ^ BsmVAD: " + str(len(set(emovad_df['Word']) & set(bsmvad_df['Word']))))   

        #open emolex
        emolex_df = pd.read_csv(os.getcwd() + '/wordlists/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',  names=['Word','Emotion','Association'], skiprows=1, sep='\t') 
        print("EmoVAD ^ EmoLEX: " + str(len(set(emovad_df['Word']) & set(emolex_df['Word']))))  

        #open mpqa
        mpqa_df = pd.read_csv(os.getcwd() + '/wordlists/MPQA_sentiment.csv',  names=['Word','Sentiment'], skiprows=0)
        print("EmoVAD ^ MPQA" + str(len(set(emovad_df['Word']) & set(mpqa_df['Word']))))

if __name__ == "__main__":
    fe = App(comment_path="/mnt/g/new_data/subset_deezer_test")
    fe.main()
