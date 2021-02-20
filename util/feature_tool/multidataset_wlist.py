from numpy.core.numeric import Infinity
import pandas as pd 

from wlist_utils import *
from string_cleaner import clean_comment
from string_cleaner import make_word_df

class MultiDataset_wlist:

    def __init__(self, wlist_paths: dict) -> None:
        columns = []
        self.comment_analysis_df = pd.DataFrame(columns=columns)
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