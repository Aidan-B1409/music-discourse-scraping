import pandas as pd 

from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df
from os import getcwd

class MultiDataset_wlist:
 
    def __init__(self, wlist_paths: dict) -> None:
        columns = []
        self.comment_analysis_df = pd.DataFrame(columns=columns)
        self.wlist_paths = wlist_paths
        self.features_wordlevel = {
            'EmoVad_^_EmoLex_glob_positive_words_v_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_v_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std_uniq': 0.0,

            "EmoVAD_^_EmoLex_glob_positive_max_word_v": 0.0, "EmoVAD_^_EmoLex_glob_positive_max_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_positive_max_word_d": 0.0, 
            
            "EmoVAD_^_EmoLex_glob_positive_min_word_v": 0.0,
            "EmoVAD_^_EmoLex_glob_positive_min_word_a": 0.0, "EmoVAD_^_EmoLex_glob_positive_min_word_d": 0.0,

            "EmoVAD_^_EmoLex_glob_positive_most_word_v": 0.0, "EmoVAD_^_EmoLex_glob_positive_most_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_positive_most_word_d": 0.0,

            'EmoVad_^_EmoLex_glob_positive_words_v_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_v_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std_uniq': 0.0,

            "EmoVAD_^_EmoLex_glob_negative_max_word_v": 0.0, "EmoVAD_^_EmoLex_glob_negative_max_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_negative_max_word_d": 0.0, 
            
            "EmoVAD_^_EmoLex_glob_negative_min_word_v": 0.0,
            "EmoVAD_^_EmoLex_glob_negative_min_word_a": 0.0, "EmoVAD_^_EmoLex_glob_negative_min_word_d": 0.0,

            "EmoVAD_^_EmoLex_glob_negative_most_word_v": 0.0, "EmoVAD_^_EmoLex_glob_negative_most_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_negative_most_word_d": 0.0, 

            'EmoVad_^_EmoLex_glob_posneg_v_ratio': 0.0, 'EmoVad_^_EmoLex_glob_posneg_a_ratio': 0.0, 
            'EmoVad_^_EmoLex_glob_posneg_d_ratio': 0.0, 

            'EmoVad_^_EmoLex_glob_posneg_v_ratio_uniq': 0.0, 'EmoVad_^_EmoLex_glob_posneg_a_ratio_uniq': 0.0, 
            'EmoVad_^_EmoLex_glob_posneg_d_ratio_uniq': 0.0, 



            'EmoVad_^_MPQA_glob_positive_words_v_mean': 0.0, 'EmoVad_^_MPQA_glob_positive_words_v_std': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_a_mean': 0.0, 'EmoVad_^_MPQA_glob_positive_words_a_std': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_d_mean': 0.0, 'EmoVad_^_MPQA_glob_positive_words_d_std': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_v_mean_uniq': 0.0, 'EmoVad_^_MPQA_glob_positive_words_v_std_uniq': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_a_mean_uniq': 0.0, 'EmoVad_^_MPQA_glob_positive_words_a_std_uniq': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_d_mean_uniq': 0.0, 'EmoVad_^_MPQA_glob_positive_words_d_std_uniq': 0.0,

            "EmoVAD_^_MPQA_glob_positive_max_word_v": 0.0, "EmoVAD_^_MPQA_glob_positive_max_word_a": 0.0,
            "EmoVAD_^_MPQA_glob_positive_max_word_d": 0.0, 
            
            "EmoVAD_^_MPQA_glob_positive_min_word_v": 0.0,
            "EmoVAD_^_MPQA_glob_positive_min_word_a": 0.0, "EmoVAD_^_MPQA_glob_positive_min_word_d": 0.0,

            "EmoVAD_^_MPQA_glob_positive_most_word_v": 0.0, "EmoVAD_^_MPQA_glob_positive_most_word_a": 0.0,
            "EmoVAD_^_MPQA_glob_positive_most_word_d": 0.0,

            'EmoVad_^_MPQA_glob_positive_words_v_mean': 0.0, 'EmoVad_^_MPQA_glob_positive_words_v_std': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_a_mean': 0.0, 'EmoVad_^_MPQA_glob_positive_words_a_std': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_d_mean': 0.0, 'EmoVad_^_MPQA_glob_positive_words_d_std': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_v_mean_uniq': 0.0, 'EmoVad_^_MPQA_glob_positive_words_v_std_uniq': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_a_mean_uniq': 0.0, 'EmoVad_^_MPQA_glob_positive_words_a_std_uniq': 0.0,
            'EmoVad_^_MPQA_glob_positive_words_d_mean_uniq': 0.0, 'EmoVad_^_MPQA_glob_positive_words_d_std_uniq': 0.0,

            "EmoVAD_^_MPQA_glob_negative_max_word_v": 0.0, "EmoVAD_^_MPQA_glob_negative_max_word_a": 0.0,
            "EmoVAD_^_MPQA_glob_negative_max_word_d": 0.0, 
            
            "EmoVAD_^_MPQA_glob_negative_min_word_v": 0.0,
            "EmoVAD_^_MPQA_glob_negative_min_word_a": 0.0, "EmoVAD_^_MPQA_glob_negative_min_word_d": 0.0,

            "EmoVAD_^_MPQA_glob_negative_most_word_v": 0.0, "EmoVAD_^_MPQA_glob_negative_most_word_a": 0.0,
            "EmoVAD_^_MPQA_glob_negative_most_word_d": 0.0, 

            'EmoVad_^_MPQA_glob_posneg_v_ratio': 0.0, 'EmoVad_^_MPQA_glob_posneg_a_ratio': 0.0, 
            'EmoVad_^_MPQA_glob_posneg_d_ratio': 0.0, 

            'EmoVad_^_MPQA_glob_posneg_v_ratio_uniq': 0.0, 'EmoVad_^_MPQA_glob_posneg_a_ratio_uniq': 0.0, 
            'EmoVad_^_MPQA_glob_posneg_d_ratio_uniq': 0.0, 


            'EmoVad_^_EmoLex_glob_positive_words_v_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_v_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std_uniq': 0.0,

            "EmoVAD_^_EmoLex_glob_positive_max_word_v": 0.0, "EmoVAD_^_EmoLex_glob_positive_max_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_positive_max_word_d": 0.0, 
            
            "EmoVAD_^_EmoLex_glob_positive_min_word_v": 0.0,
            "EmoVAD_^_EmoLex_glob_positive_min_word_a": 0.0, "EmoVAD_^_EmoLex_glob_positive_min_word_d": 0.0,

            "EmoVAD_^_EmoLex_glob_positive_most_word_v": 0.0, "EmoVAD_^_EmoLex_glob_positive_most_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_positive_most_word_d": 0.0,

            'EmoVad_^_EmoLex_glob_positive_words_v_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_v_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_v_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_a_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_a_std_uniq': 0.0,
            'EmoVad_^_EmoLex_glob_positive_words_d_mean_uniq': 0.0, 'EmoVad_^_EmoLex_glob_positive_words_d_std_uniq': 0.0,

            "EmoVAD_^_EmoLex_glob_negative_max_word_v": 0.0, "EmoVAD_^_EmoLex_glob_negative_max_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_negative_max_word_d": 0.0, 
            
            "EmoVAD_^_EmoLex_glob_negative_min_word_v": 0.0,
            "EmoVAD_^_EmoLex_glob_negative_min_word_a": 0.0, "EmoVAD_^_EmoLex_glob_negative_min_word_d": 0.0,

            "EmoVAD_^_EmoLex_glob_negative_most_word_v": 0.0, "EmoVAD_^_EmoLex_glob_negative_most_word_a": 0.0,
            "EmoVAD_^_EmoLex_glob_negative_most_word_d": 0.0, 

            'EmoVad_^_EmoLex_glob_posneg_v_ratio': 0.0, 'EmoVad_^_EmoLex_glob_posneg_a_ratio': 0.0, 
            'EmoVad_^_EmoLex_glob_posneg_d_ratio': 0.0, 

            'EmoVad_^_EmoLex_glob_posneg_v_ratio_uniq': 0.0, 'EmoVad_^_EmoLex_glob_posneg_a_ratio_uniq': 0.0, 
            'EmoVad_^_EmoLex_glob_posneg_d_ratio_uniq': 0.0, 


            'BsmVad_^_EmoLex_glob_positive_words_v_mean': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_v_std': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_a_mean': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_a_std': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_d_mean': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_d_std': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_v_mean_uniq': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_v_std_uniq': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_a_mean_uniq': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_a_std_uniq': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_d_mean_uniq': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_d_std_uniq': 0.0,

            "BsmVad_^_EmoLex_glob_positive_max_word_v": 0.0, "BsmVad_^_EmoLex_glob_positive_max_word_a": 0.0,
            "BsmVad_^_EmoLex_glob_positive_max_word_d": 0.0, 
            
            "BsmVad_^_EmoLex_glob_positive_min_word_v": 0.0,
            "BsmVad_^_EmoLex_glob_positive_min_word_a": 0.0, "BsmVad_^_EmoLex_glob_positive_min_word_d": 0.0,

            "BsmVad_^_EmoLex_glob_positive_most_word_v": 0.0, "BsmVad_^_EmoLex_glob_positive_most_word_a": 0.0,
            "BsmVad_^_EmoLex_glob_positive_most_word_d": 0.0,

            'BsmVad_^_EmoLex_glob_positive_words_v_mean': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_v_std': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_a_mean': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_a_std': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_d_mean': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_d_std': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_v_mean_uniq': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_v_std_uniq': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_a_mean_uniq': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_a_std_uniq': 0.0,
            'BsmVad_^_EmoLex_glob_positive_words_d_mean_uniq': 0.0, 'BsmVad_^_EmoLex_glob_positive_words_d_std_uniq': 0.0,

            "BsmVad_^_EmoLex_glob_negative_max_word_v": 0.0, "BsmVad_^_EmoLex_glob_negative_max_word_a": 0.0,
            "BsmVad_^_EmoLex_glob_negative_max_word_d": 0.0, 
            
            "BsmVad_^_EmoLex_glob_negative_min_word_v": 0.0,
            "BsmVad_^_EmoLex_glob_negative_min_word_a": 0.0, "BsmVad_^_EmoLex_glob_negative_min_word_d": 0.0,

            "BsmVad_^_EmoLex_glob_negative_most_word_v": 0.0, "BsmVad_^_EmoLex_glob_negative_most_word_a": 0.0,
            "BsmVad_^_EmoLex_glob_negative_most_word_d": 0.0, 

            'BsmVad_^_EmoLex_glob_posneg_v_ratio': 0.0, 'BsmVad_^_EmoLex_glob_posneg_a_ratio': 0.0, 
            'BsmVad_^_EmoLex_glob_posneg_d_ratio': 0.0, 

            'BsmVad_^_EmoLex_glob_posneg_v_ratio_uniq': 0.0, 'BsmVad_^_EmoLex_glob_posneg_a_ratio_uniq': 0.0, 
            'BsmVad_^_EmoLex_glob_posneg_d_ratio_uniq': 0.0, 


            'BsmVad_^_MPQA_glob_positive_words_v_mean': 0.0, 'BsmVad_^_MPQA_glob_positive_words_v_std': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_a_mean': 0.0, 'BsmVad_^_MPQA_glob_positive_words_a_std': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_d_mean': 0.0, 'BsmVad_^_MPQA_glob_positive_words_d_std': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_v_mean_uniq': 0.0, 'BsmVad_^_MPQA_glob_positive_words_v_std_uniq': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_a_mean_uniq': 0.0, 'BsmVad_^_MPQA_glob_positive_words_a_std_uniq': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_d_mean_uniq': 0.0, 'BsmVad_^_MPQA_glob_positive_words_d_std_uniq': 0.0,

            "BsmVad_^_MPQA_glob_positive_max_word_v": 0.0, "BsmVad_^_MPQA_glob_positive_max_word_a": 0.0,
            "BsmVad_^_MPQA_glob_positive_max_word_d": 0.0, 
            
            "BsmVad_^_MPQA_glob_positive_min_word_v": 0.0,
            "BsmVad_^_MPQA_glob_positive_min_word_a": 0.0, "BsmVad_^_MPQA_glob_positive_min_word_d": 0.0,

            "BsmVad_^_MPQA_glob_positive_most_word_v": 0.0, "BsmVad_^_MPQA_glob_positive_most_word_a": 0.0,
            "BsmVad_^_MPQA_glob_positive_most_word_d": 0.0,

            'BsmVad_^_MPQA_glob_positive_words_v_mean': 0.0, 'BsmVad_^_MPQA_glob_positive_words_v_std': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_a_mean': 0.0, 'BsmVad_^_MPQA_glob_positive_words_a_std': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_d_mean': 0.0, 'BsmVad_^_MPQA_glob_positive_words_d_std': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_v_mean_uniq': 0.0, 'BsmVad_^_MPQA_glob_positive_words_v_std_uniq': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_a_mean_uniq': 0.0, 'BsmVad_^_MPQA_glob_positive_words_a_std_uniq': 0.0,
            'BsmVad_^_MPQA_glob_positive_words_d_mean_uniq': 0.0, 'BsmVad_^_MPQA_glob_positive_words_d_std_uniq': 0.0,

            "BsmVad_^_MPQA_glob_negative_max_word_v": 0.0, "BsmVad_^_MPQA_glob_negative_max_word_a": 0.0,
            "BsmVad_^_MPQA_glob_negative_max_word_d": 0.0, 
            
            "BsmVad_^_MPQA_glob_negative_min_word_v": 0.0,
            "BsmVad_^_MPQA_glob_negative_min_word_a": 0.0, "BsmVad_^_MPQA_glob_negative_min_word_d": 0.0,

            "BsmVad_^_MPQA_glob_negative_most_word_v": 0.0, "BsmVad_^_MPQA_glob_negative_most_word_a": 0.0,
            "BsmVad_^_MPQA_glob_negative_most_word_d": 0.0, 

            'BsmVad_^_MPQA_glob_posneg_v_ratio': 0.0, 'BsmVad_^_MPQA_glob_posneg_a_ratio': 0.0, 
            'BsmVad_^_MPQA_glob_posneg_d_ratio': 0.0, 

            'BsmVad_^_MPQA_glob_posneg_v_ratio_uniq': 0.0, 'BsmVad_^_MPQA_glob_posneg_a_ratio_uniq': 0.0, 
            'BsmVad_^_MPQA_glob_posneg_d_ratio_uniq': 0.0, 
            
        }
        self.features_commentlevel = {

            'EmoVad_^_EmoLex_positive_words_v_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_stds_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_std_uniqs_std': 0.0,

            "EmoVAD_^_EmoLex_positive_max_word_v_mean": 0.0, "EmoVAD_^_EmoLex_positive_max_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_positive_max_word_a_mean": 0.0, "EmoVAD_^_EmoLex_positive_max_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_positive_max_word_d_mean": 0.0, "EmoVAD_^_EmoLex_positive_max_word_d_std": 0.0, 
            
            "EmoVAD_^_EmoLex_positive_min_word_v_mean": 0.0, "EmoVAD_^_EmoLex_positive_min_word_v_std": 0.0,
            "EmoVAD_^_EmoLex_positive_min_word_a_mean": 0.0, "EmoVAD_^_EmoLex_positive_min_word_a_std": 0.0, 
            "EmoVAD_^_EmoLex_positive_min_word_d_mean": 0.0, "EmoVAD_^_EmoLex_positive_min_word_d_std": 0.0,

            "EmoVAD_^_EmoLex_positive_most_word_v_mean": 0.0, "EmoVAD_^_EmoLex_positive_most_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_positive_most_word_a_mean": 0.0, "EmoVAD_^_EmoLex_positive_most_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_positive_most_word_d_mean": 0.0, "EmoVAD_^_EmoLex_positive_most_word_d_std": 0.0,

            'EmoVad_^_EmoLex_positive_words_v_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_a_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_std_uniqs_std': 0.0,

            "EmoVAD_^_EmoLex_negative_max_word_v_mean": 0.0, "EmoVAD_^_EmoLex_negative_max_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_negative_max_word_a_mean": 0.0, "EmoVAD_^_EmoLex_negative_max_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_negative_max_word_d_mean": 0.0, "EmoVAD_^_EmoLex_negative_max_word_d_std": 0.0, 
            
            "EmoVAD_^_EmoLex_negative_min_word_v_mean": 0.0, "EmoVAD_^_EmoLex_negative_min_word_v_std": 0.0,
            "EmoVAD_^_EmoLex_negative_min_word_a_mean": 0.0, "EmoVAD_^_EmoLex_negative_min_word_a_std": 0.0, 
            "EmoVAD_^_EmoLex_negative_min_word_d_mean": 0.0, "EmoVAD_^_EmoLex_negative_min_word_d_std": 0.0,

            "EmoVAD_^_EmoLex_negative_most_word_v_mean": 0.0, "EmoVAD_^_EmoLex_negative_most_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_negative_most_word_a_mean": 0.0, "EmoVAD_^_EmoLex_negative_most_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_negative_most_word_d_mean": 0.0, "EmoVAD_^_EmoLex_negative_most_word_d_std": 0.0, 

            'EmoVad_^_EmoLex_posneg_v_ratios_mean': 0.0, 'EmoVad_^_EmoLex_posneg_v_ratios_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_a_ratios_mean': 0.0, 'EmoVad_^_EmoLex_posneg_a_ratios_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_d_ratios_mean': 0.0, 'EmoVad_^_EmoLex_posneg_d_ratios_std': 0.0, 

            'EmoVad_^_EmoLex_posneg_v_ratio_uniq_mean': 0.0, 'EmoVad_^_EmoLex_posneg_v_ratio_uniq_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_a_ratio_uniq_mean': 0.0, 'EmoVad_^_EmoLex_posneg_a_ratio_uniq_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_d_ratio_uniq_mean': 0.0, 'EmoVad_^_EmoLex_posneg_d_ratio_uniq_std': 0.0, 



            'EmoVad_^_MPQA_positive_words_v_means_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_means_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_v_stds_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_stds_std': 0.0,
            'EmoVad_^_MPQA_positive_words_a_means_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_means_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_a_stds_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_stds_std': 0.0,
            'EmoVad_^_MPQA_positive_words_d_means_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_means_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_d_stds_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_stds_std': 0.0,
            'EmoVad_^_MPQA_positive_words_v_mean_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_mean_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_v_std_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_std_uniqs_std': 0.0,
            'EmoVad_^_MPQA_positive_words_a_mean_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_mean_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_a_std_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_std_uniqs_std': 0.0,
            'EmoVad_^_MPQA_positive_words_d_mean_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_mean_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_d_std_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_std_uniqs_std': 0.0,

            "EmoVAD_^_MPQA_positive_max_word_v_mean": 0.0, "EmoVAD_^_MPQA_positive_max_word_v_std": 0.0, 
            "EmoVAD_^_MPQA_positive_max_word_a_mean": 0.0, "EmoVAD_^_MPQA_positive_max_word_a_std": 0.0,
            "EmoVAD_^_MPQA_positive_max_word_d_mean": 0.0, "EmoVAD_^_MPQA_positive_max_word_d_std": 0.0, 
            
            "EmoVAD_^_MPQA_positive_min_word_v_mean": 0.0, "EmoVAD_^_MPQA_positive_min_word_v_std": 0.0,
            "EmoVAD_^_MPQA_positive_min_word_a_mean": 0.0, "EmoVAD_^_MPQA_positive_min_word_a_std": 0.0, 
            "EmoVAD_^_MPQA_positive_min_word_d_mean": 0.0, "EmoVAD_^_MPQA_positive_min_word_d_std": 0.0,

            "EmoVAD_^_MPQA_positive_most_word_v_mean": 0.0, "EmoVAD_^_MPQA_positive_most_word_v_std": 0.0, 
            "EmoVAD_^_MPQA_positive_most_word_a_mean": 0.0, "EmoVAD_^_MPQA_positive_most_word_a_std": 0.0,
            "EmoVAD_^_MPQA_positive_most_word_d_mean": 0.0, "EmoVAD_^_MPQA_positive_most_word_d_std": 0.0,

            'EmoVad_^_MPQA_positive_words_v_means_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_means_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_v_stds_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_stds_std': 0.0,
            'EmoVad_^_MPQA_positive_words_a_means_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_means_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_a_stds_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_stds_std': 0.0,
            'EmoVad_^_MPQA_positive_words_d_means_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_means_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_d_stds_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_stds_std': 0.0,
            'EmoVad_^_MPQA_positive_words_v_mean_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_mean_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_v_std_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_v_std_uniqs_std': 0.0,
            'EmoVad_^_MPQA_positive_words_a_mean_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_mean_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_a_std_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_a_std_uniqs_std': 0.0,
            'EmoVad_^_MPQA_positive_words_d_mean_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_mean_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_positive_words_d_std_uniqs_mean': 0.0, 'EmoVad_^_MPQA_positive_words_d_std_uniqs_std': 0.0,

            "EmoVAD_^_MPQA_negative_max_word_v_mean": 0.0, "EmoVAD_^_MPQA_negative_max_word_v_std": 0.0, 
            "EmoVAD_^_MPQA_negative_max_word_a_mean": 0.0, "EmoVAD_^_MPQA_negative_max_word_a_std": 0.0,
            "EmoVAD_^_MPQA_negative_max_word_d_mean": 0.0, "EmoVAD_^_MPQA_negative_max_word_d_std": 0.0, 
            
            "EmoVAD_^_MPQA_negative_min_word_v_mean": 0.0, "EmoVAD_^_MPQA_negative_min_word_v_std": 0.0,
            "EmoVAD_^_MPQA_negative_min_word_a_mean": 0.0, "EmoVAD_^_MPQA_negative_min_word_a_std": 0.0, 
            "EmoVAD_^_MPQA_negative_min_word_d_mean": 0.0, "EmoVAD_^_MPQA_negative_min_word_d_std": 0.0,

            "EmoVAD_^_MPQA_negative_most_word_v_mean": 0.0, "EmoVAD_^_MPQA_negative_most_word_v_std": 0.0, 
            "EmoVAD_^_MPQA_negative_most_word_a_mean": 0.0, "EmoVAD_^_MPQA_negative_most_word_a_std": 0.0,
            "EmoVAD_^_MPQA_negative_most_word_d_mean": 0.0, "EmoVAD_^_MPQA_negative_most_word_d_std": 0.0, 

            'EmoVad_^_MPQA_posneg_v_ratios_mean': 0.0, 'EmoVad_^_MPQA_posneg_v_ratios_std': 0.0, 
            'EmoVad_^_MPQA_posneg_a_ratios_mean': 0.0, 'EmoVad_^_MPQA_posneg_a_ratios_std': 0.0, 
            'EmoVad_^_MPQA_posneg_d_ratios_mean': 0.0, 'EmoVad_^_MPQA_posneg_d_ratios_std': 0.0, 

            'EmoVad_^_MPQA_posneg_v_ratio_uniqs_mean': 0.0, 'EmoVad_^_MPQA_posneg_v_ratio_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_posneg_a_ratio_uniqs_mean': 0.0, 'EmoVad_^_MPQA_posneg_a_ratio_uniqs_std': 0.0, 
            'EmoVad_^_MPQA_posneg_d_ratio_uniqs_mean': 0.0, 'EmoVad_^_MPQA_posneg_d_ratio_uniqs_std': 0.0, 


            'EmoVad_^_EmoLex_positive_words_v_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_a_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_std_uniqs_std': 0.0,

            "EmoVAD_^_EmoLex_positive_max_word_v_mean": 0.0, "EmoVAD_^_EmoLex_positive_max_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_positive_max_word_a_mean": 0.0, "EmoVAD_^_EmoLex_positive_max_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_positive_max_word_d_mean": 0.0, "EmoVAD_^_EmoLex_positive_max_word_d_std": 0.0, 
            
            "EmoVAD_^_EmoLex_positive_min_word_v_mean": 0.0, "EmoVAD_^_EmoLex_positive_min_word_v_std": 0.0,
            "EmoVAD_^_EmoLex_positive_min_word_a_mean": 0.0, "EmoVAD_^_EmoLex_positive_min_word_a_std": 0.0, 
            "EmoVAD_^_EmoLex_positive_min_word_d_mean": 0.0, "EmoVAD_^_EmoLex_positive_min_word_d_std": 0.0,

            "EmoVAD_^_EmoLex_positive_most_word_v_mean": 0.0, "EmoVAD_^_EmoLex_positive_most_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_positive_most_word_a_mean": 0.0, "EmoVAD_^_EmoLex_positive_most_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_positive_most_word_d_mean": 0.0, "EmoVAD_^_EmoLex_positive_most_word_d_std": 0.0,

            'EmoVad_^_EmoLex_positive_words_v_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_a_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_means_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_means_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_stds_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_stds_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_v_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_v_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_a_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_a_std_uniqs_std': 0.0,
            'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_mean_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_positive_words_d_std_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_positive_words_d_std_uniqs_std': 0.0,

            "EmoVAD_^_EmoLex_negative_max_word_v_mean": 0.0, "EmoVAD_^_EmoLex_negative_max_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_negative_max_word_a_mean": 0.0, "EmoVAD_^_EmoLex_negative_max_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_negative_max_word_d_mean": 0.0, "EmoVAD_^_EmoLex_negative_max_word_d_std": 0.0, 
            
            "EmoVAD_^_EmoLex_negative_min_word_v_mean": 0.0, "EmoVAD_^_EmoLex_negative_min_word_v_std": 0.0,
            "EmoVAD_^_EmoLex_negative_min_word_a_mean": 0.0, "EmoVAD_^_EmoLex_negative_min_word_a_std": 0.0, 
            "EmoVAD_^_EmoLex_negative_min_word_d_mean": 0.0, "EmoVAD_^_EmoLex_negative_min_word_d_std": 0.0,

            "EmoVAD_^_EmoLex_negative_most_word_v_mean": 0.0, "EmoVAD_^_EmoLex_negative_most_word_v_std": 0.0, 
            "EmoVAD_^_EmoLex_negative_most_word_a_mean": 0.0, "EmoVAD_^_EmoLex_negative_most_word_a_std": 0.0,
            "EmoVAD_^_EmoLex_negative_most_word_d_mean": 0.0, "EmoVAD_^_EmoLex_negative_most_word_d_std": 0.0, 

            'EmoVad_^_EmoLex_posneg_v_ratios_mean': 0.0, 'EmoVad_^_EmoLex_posneg_v_ratios_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_a_ratios_mean': 0.0, 'EmoVad_^_EmoLex_posneg_a_ratios_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_d_ratios_mean': 0.0, 'EmoVad_^_EmoLex_posneg_d_ratios_std': 0.0, 

            'EmoVad_^_EmoLex_posneg_v_ratio_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_posneg_v_ratio_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_a_ratio_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_posneg_a_ratio_uniqs_std': 0.0, 
            'EmoVad_^_EmoLex_posneg_d_ratio_uniqs_mean': 0.0, 'EmoVad_^_EmoLex_posneg_d_ratio_uniqs_std': 0.0, 


            'BsmVad_^_EmoLex_positive_words_v_means_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_means_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_v_stds_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_stds_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_a_means_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_means_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_a_stds_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_stds_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_d_means_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_means_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_d_stds_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_stds_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_v_mean_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_mean_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_v_std_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_std_uniqs_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_a_mean_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_mean_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_a_std_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_std_uniqs_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_d_mean_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_mean_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_d_std_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_std_uniqs_std': 0.0,

            "BsmVad_^_EmoLex_positive_max_word_v_mean": 0.0, "BsmVad_^_EmoLex_positive_max_word_v_std": 0.0, 
            "BsmVad_^_EmoLex_positive_max_word_a_mean": 0.0, "BsmVad_^_EmoLex_positive_max_word_a_std": 0.0,
            "BsmVad_^_EmoLex_positive_max_word_d_mean": 0.0, "BsmVad_^_EmoLex_positive_max_word_d_std": 0.0, 
            
            "BsmVad_^_EmoLex_positive_min_word_v_mean": 0.0, "BsmVad_^_EmoLex_positive_min_word_v_std": 0.0,
            "BsmVad_^_EmoLex_positive_min_word_a_mean": 0.0, "BsmVad_^_EmoLex_positive_min_word_a_std": 0.0, 
            "BsmVad_^_EmoLex_positive_min_word_d_mean": 0.0, "BsmVad_^_EmoLex_positive_min_word_d_std": 0.0,

            "BsmVad_^_EmoLex_positive_most_word_v_mean": 0.0, "BsmVad_^_EmoLex_positive_most_word_v_std": 0.0, 
            "BsmVad_^_EmoLex_positive_most_word_a_mean": 0.0, "BsmVad_^_EmoLex_positive_most_word_a_std": 0.0,
            "BsmVad_^_EmoLex_positive_most_word_d_mean": 0.0, "BsmVad_^_EmoLex_positive_most_word_d_std": 0.0,

            'BsmVad_^_EmoLex_positive_words_v_means_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_means_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_v_stds_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_stds_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_a_means_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_means_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_a_stds_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_stds_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_d_means_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_means_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_d_stds_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_stds_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_v_mean_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_mean_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_v_std_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_v_std_uniqs_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_a_mean_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_mean_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_a_std_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_a_std_uniqs_std': 0.0,
            'BsmVad_^_EmoLex_positive_words_d_mean_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_mean_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_positive_words_d_std_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_positive_words_d_std_uniqs_std': 0.0,

            "BsmVad_^_EmoLex_negative_max_word_v_mean": 0.0, "BsmVad_^_EmoLex_negative_max_word_v_std": 0.0, 
            "BsmVad_^_EmoLex_negative_max_word_a_mean": 0.0, "BsmVad_^_EmoLex_negative_max_word_a_std": 0.0,
            "BsmVad_^_EmoLex_negative_max_word_d_mean": 0.0, "BsmVad_^_EmoLex_negative_max_word_d_std": 0.0, 
            
            "BsmVad_^_EmoLex_negative_min_word_v_mean": 0.0, "BsmVad_^_EmoLex_negative_min_word_v_std": 0.0,
            "BsmVad_^_EmoLex_negative_min_word_a_mean": 0.0, "BsmVad_^_EmoLex_negative_min_word_a_std": 0.0, 
            "BsmVad_^_EmoLex_negative_min_word_d_mean": 0.0, "BsmVad_^_EmoLex_negative_min_word_d_std": 0.0,

            "BsmVad_^_EmoLex_negative_most_word_v_mean": 0.0, "BsmVad_^_EmoLex_negative_most_word_v_std": 0.0, 
            "BsmVad_^_EmoLex_negative_most_word_a_mean": 0.0, "BsmVad_^_EmoLex_negative_most_word_a_std": 0.0,
            "BsmVad_^_EmoLex_negative_most_word_d_mean": 0.0, "BsmVad_^_EmoLex_negative_most_word_d_std": 0.0, 

            'BsmVad_^_EmoLex_posneg_v_ratios_mean': 0.0, 'BsmVad_^_EmoLex_posneg_v_ratios_std': 0.0, 
            'BsmVad_^_EmoLex_posneg_a_ratios_mean': 0.0, 'BsmVad_^_EmoLex_posneg_a_ratios_std': 0.0, 
            'BsmVad_^_EmoLex_posneg_d_ratios_mean': 0.0, 'BsmVad_^_EmoLex_posneg_d_ratios_std': 0.0, 

            'BsmVad_^_EmoLex_posneg_v_ratio_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_posneg_v_ratio_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_posneg_a_ratio_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_posneg_a_ratio_uniqs_std': 0.0, 
            'BsmVad_^_EmoLex_posneg_d_ratio_uniqs_mean': 0.0, 'BsmVad_^_EmoLex_posneg_d_ratio_uniqs_std': 0.0, 


            'BsmVad_^_MPQA_positive_words_v_means_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_means_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_v_stds_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_stds_std': 0.0,
            'BsmVad_^_MPQA_positive_words_a_means_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_means_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_a_stds_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_stds_std': 0.0,
            'BsmVad_^_MPQA_positive_words_d_means_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_means_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_d_stds_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_stds_std': 0.0,
            'BsmVad_^_MPQA_positive_words_v_mean_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_mean_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_v_std_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_std_uniqs_std': 0.0,
            'BsmVad_^_MPQA_positive_words_a_mean_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_mean_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_a_std_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_std_uniqs_std': 0.0,
            'BsmVad_^_MPQA_positive_words_d_mean_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_mean_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_d_std_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_std_uniqs_std': 0.0,

            "BsmVad_^_MPQA_positive_max_word_v_mean": 0.0, "BsmVad_^_MPQA_positive_max_word_v_std": 0.0, 
            "BsmVad_^_MPQA_positive_max_word_a_mean": 0.0, "BsmVad_^_MPQA_positive_max_word_a_std": 0.0,
            "BsmVad_^_MPQA_positive_max_word_d_mean": 0.0, "BsmVad_^_MPQA_positive_max_word_d_std": 0.0, 
            
            "BsmVad_^_MPQA_positive_min_word_v_mean": 0.0, "BsmVad_^_MPQA_positive_min_word_v_std": 0.0,
            "BsmVad_^_MPQA_positive_min_word_a_mean": 0.0, "BsmVad_^_MPQA_positive_min_word_a_std": 0.0, 
            "BsmVad_^_MPQA_positive_min_word_d_mean": 0.0, "BsmVad_^_MPQA_positive_min_word_d_std": 0.0,

            "BsmVad_^_MPQA_positive_most_word_v_mean": 0.0, "BsmVad_^_MPQA_positive_most_word_v_std": 0.0, 
            "BsmVad_^_MPQA_positive_most_word_a_mean": 0.0, "BsmVad_^_MPQA_positive_most_word_a_std": 0.0,
            "BsmVad_^_MPQA_positive_most_word_d_mean": 0.0, "BsmVad_^_MPQA_positive_most_word_d_std": 0.0,

            'BsmVad_^_MPQA_positive_words_v_means_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_means_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_v_stds_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_stds_std': 0.0,
            'BsmVad_^_MPQA_positive_words_a_means_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_means_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_a_stds_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_stds_std': 0.0,
            'BsmVad_^_MPQA_positive_words_d_means_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_means_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_d_stds_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_stds_std': 0.0,
            'BsmVad_^_MPQA_positive_words_v_mean_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_mean_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_v_std_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_v_std_uniqs_std': 0.0,
            'BsmVad_^_MPQA_positive_words_a_mean_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_mean_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_a_std_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_a_std_uniqs_std': 0.0,
            'BsmVad_^_MPQA_positive_words_d_mean_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_mean_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_positive_words_d_std_uniqs_mean': 0.0, 'BsmVad_^_MPQA_positive_words_d_std_uniqs_std': 0.0,

            "BsmVad_^_MPQA_negative_max_word_v_mean": 0.0, "BsmVad_^_MPQA_negative_max_word_v_std": 0.0, 
            "BsmVad_^_MPQA_negative_max_word_a_mean": 0.0, "BsmVad_^_MPQA_negative_max_word_a_std": 0.0,
            "BsmVad_^_MPQA_negative_max_word_d_mean": 0.0, "BsmVad_^_MPQA_negative_max_word_d_std": 0.0, 
            
            "BsmVad_^_MPQA_negative_min_word_v_mean": 0.0, "BsmVad_^_MPQA_negative_min_word_v_std": 0.0,
            "BsmVad_^_MPQA_negative_min_word_a_mean": 0.0, "BsmVad_^_MPQA_negative_min_word_a_std": 0.0, 
            "BsmVad_^_MPQA_negative_min_word_d_mean": 0.0, "BsmVad_^_MPQA_negative_min_word_d_std": 0.0,

            "BsmVad_^_MPQA_negative_most_word_v_mean": 0.0, "BsmVad_^_MPQA_negative_most_word_v_std": 0.0, 
            "BsmVad_^_MPQA_negative_most_word_a_mean": 0.0, "BsmVad_^_MPQA_negative_most_word_a_std": 0.0,
            "BsmVad_^_MPQA_negative_most_word_d_mean": 0.0, "BsmVad_^_MPQA_negative_most_word_d_std": 0.0, 

            'BsmVad_^_MPQA_posneg_v_ratios_mean': 0.0, 'BsmVad_^_MPQA_posneg_v_ratios_std': 0.0, 
            'BsmVad_^_MPQA_posneg_a_ratios_mean': 0.0, 'BsmVad_^_MPQA_posneg_a_ratios_std': 0.0, 
            'BsmVad_^_MPQA_posneg_d_ratios_mean': 0.0, 'BsmVad_^_MPQA_posneg_d_ratios_std': 0.0, 

            'BsmVad_^_MPQA_posneg_v_ratio_uniqs_mean': 0.0, 'BsmVad_^_MPQA_posneg_v_ratio_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_posneg_a_ratio_uniqs_mean': 0.0, 'BsmVad_^_MPQA_posneg_a_ratio_uniqs_std': 0.0, 
            'BsmVad_^_MPQA_posneg_d_ratio_uniqs_mean': 0.0, 'BsmVad_^_MPQA_posneg_d_ratio_uniqs_std': 0.0, 
        }

    def wordlevel_analysis(self, song_df, glob_df) -> dict:
        if(len(song_df) > 0):
            
            # EmoVAD ^ EmoLex
            emovad_df = pd.read_csv(self._getpath('EmoVAD'), names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t')
            emolex_df = pd.read_csv(self._getpath('EmoLex'), names=['Word','Emotion','Association'], skiprows=1, sep='\t')
            
            emolex_positive_words = self._get_affect_subset('positive', emolex_df)
            emolex_negative_words = self._get_affect_subset('negative', emolex_df)
            emovad_emolex_positive_words = pd.merge(emolex_positive_words, emovad_df, on='Word')
            emovad_emolex_negative_words = pd.merge(emolex_negative_words, emovad_df, on='Word')

            found_emovad_emolex_positive_words_df = unsquished_intersection(song_df, emovad_emolex_positive_words)
            found_emovad_emolex_negative_words_df = unsquished_intersection(song_df, emovad_emolex_negative_words)
            found_emovad_emolex_positive_words_uniq_df = glob_intersection(glob_df, emovad_emolex_positive_words)
            found_emovad_emolex_negative_words_uniq_df = glob_intersection(glob_df, emovad_emolex_negative_words)

            # EmoVAD ^ MPQA
            mpqa_df = pd.read_csv(self._getpath('MPQA'),  names=['Word','Sentiment'], skiprows=0)
            mpqa_positive_words = self._get_mpqa_sentiment_subset('positive', mpqa_df)
            mpqa_negative_words = self._get_mpqa_sentiment_subset('negative', mpqa_df)

            emovad_mpqa_positive_words = pd.merge(mpqa_positive_words, emovad_df, on='Word')
            emovad_mpqa_negative_words = pd.merge(mpqa_negative_words, emovad_df, on='Word')

            found_emovad_mpqa_positive_words_df = unsquished_intersection(song_df, emovad_mpqa_positive_words)
            found_emovad_mpqa_negative_words_df = unsquished_intersection(song_df, emovad_mpqa_negative_words)
            found_emovad_mpqa_positive_words_uniq_df = glob_intersection(glob_df, emovad_mpqa_positive_words)
            found_emovad_mpqa_negative_words_uniq_df = glob_intersection(glob_df, emovad_mpqa_negative_words)

            #BsmVAD ^ EmoLex
            bsmvad_df = pd.read_csv(self._getpath('ANEW_Extended'), encoding='utf-8', engine='python')
            # drop unneeded columns
            bsmvad_df.drop(bsmvad_df.iloc[:, 10:64].columns, axis = 1, inplace = True) 
            bsmvad_df.drop(['V.Rat.Sum', 'A.Rat.Sum','D.Rat.Sum'], axis = 1, inplace = True) 
            # drop blank rows, if any
            bsmvad_df = bsmvad_df[bsmvad_df['Word'].notnull()]

            bsmvad_emolex_positive_words = pd.merge(emolex_positive_words, bsmvad_df, on='Word')
            bsmvad_emolex_negative_words = pd.merge(emolex_negative_words, bsmvad_df, on='Word')

            found_bsmvad_emolex_positive_words_df = unsquished_intersection(song_df, bsmvad_emolex_positive_words)
            found_bsmvad_emolex_negative_words_df = unsquished_intersection(song_df, bsmvad_emolex_negative_words)
            found_bsmvad_emolex_positive_words_uniq_df = glob_intersection(glob_df, bsmvad_emolex_positive_words)
            found_bsmvad_emolex_negative_words_uniq_df = glob_intersection(glob_df, bsmvad_emolex_negative_words)

            #BsmVAD ^ MPQA 
            bsmvad_mpqa_positive_words = pd.merge(mpqa_positive_words, bsmvad_df, on='Word')
            bsmvad_mpqa_negative_words = pd.merge(mpqa_negative_words, bsmvad_df, on='Word')

            found_bsmvad_mpqa_positive_words_df = unsquished_intersection(song_df, bsmvad_mpqa_positive_words)
            found_bsmvad_mpqa_negative_words_df = unsquished_intersection(song_df, bsmvad_mpqa_negative_words)
            found_bsmvad_mpqa_positive_words_uniq_df = glob_intersection(glob_df, bsmvad_mpqa_positive_words)
            found_bsmvad_mpqa_negative_words_uniq_df = glob_intersection(glob_df, bsmvad_mpqa_negative_words)

            print("BsmVad + mpqa: " + str(len(found_bsmvad_mpqa_positive_words_df)))

            

        return self.features_wordlevel

    def process_comment(self, index, comment):
        pass

    def analyze_comments(self) -> dict:
        return self.features_commentlevel

    def _get_affect_subset(self, affect_key, emolex_df) -> pd.DataFrame:
        emolex_affect_subset = emolex_df[(emolex_df['Emotion'] == affect_key)]
        indexes = emolex_affect_subset[emolex_affect_subset['Association'] == 0].index
        emolex_affect_subset.drop(indexes, inplace=True)
        return emolex_affect_subset

    def _get_mpqa_sentiment_subset(self, affect_key, mpqa_df) -> pd.DataFrame:
        excluded_indexes = mpqa_df[mpqa_df['Sentiment'] != affect_key].index
        mpqa_subset = mpqa_df.drop(excluded_indexes)
        return mpqa_subset

    def _getpath(self, key):
        return getcwd() + '/wordlists/' + self.wlist_paths[key]