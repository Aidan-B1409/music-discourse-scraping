import pandas as pd
from string_cleaner import make_word_df
from string_cleaner import clean_comment
from statistics import mean
from statistics import stdev

class EmoVADGenerator:

    features = {
        "EmoVAD_glob_v_mean": 0.0, "EmoVAD_glob_v_stdev": 0.0,
        "EmoVAD_glob_a_mean": 0.0, "EmoVAD_glob_a_stdev": 0.0,
        "EmoVAD_glob_d_mean": 0.0, "EmoVAD_glob_d_stdev": 0.0,
        "EmoVAD_glob_v_mean_uniq": 0.0, "EmoVAD_glob_v_stdev_uniq": 0.0,
        "EmoVAD_glob_a_mean_uniq": 0.0, "EmoVAD_glob_a_stdev_uniq": 0.0,
        "EmoVAD_glob_d_mean_uniq": 0.0, "EmoVAD_glob_d_stdev_uniq": 0.0,

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

    }

    def __init__(self, song_df, glob_df, emovad_df) -> None:
        self.song_df = song_df
        self.glob_df = glob_df
        self.emovad_df = emovad_df

    def get_features(self) -> dict:
        self.glob_features()
        self.independent_features()
        return self.features

    
    # Find the features of the word glob for a specific song 
    def glob_features(self) -> None:
        semantic_word_df = self.make_intersection(self.glob_df, self.emovad_df)

        semantic_word_df.to_csv("out.csv", index=False)

        # VAD means including duplicates
        # We can't use the default mean function here, because the number of rows does not correlate to the number of total words 
        # It DOES correlate to the number of unique rows, which is why these functions are used below
        if semantic_word_df['Count'].sum() > 0: 
            v_mean = semantic_word_df['V_Total'].sum() / semantic_word_df['Count'].sum()
            self.features["EmoVAD_glob_v_mean"] = v_mean
            if(len(semantic_word_df['V_Total']) >= 2):
                self.features["EmoVAD_glob_v_stdev"] = stdev(semantic_word_df['V_Total'], xbar = v_mean)

            a_mean = semantic_word_df['A_Total'].sum() / semantic_word_df['Count'].sum()
            self.features["EmoVAD_glob_a_mean"] = a_mean
            if(len(semantic_word_df['A_Total']) >= 2):
                self.features["EmoVAD_glob_a_stdev"] = stdev(semantic_word_df['A_Total'], xbar = a_mean)

            d_mean = semantic_word_df['D_Total'].sum() / semantic_word_df['Count'].sum()
            self.features["EmoVAD_glob_d_mean"] = d_mean
            if(len(semantic_word_df['D_Total']) >= 2):
                self.features["EmoVAD_glob_d_stdev"] = stdev(semantic_word_df['D_Total'], xbar = d_mean)

        #VAD means on unique words only
        self.features["EmoVAD_glob_v_mean_uniq"] = semantic_word_df['Valence'].mean()
        self.features["EmoVAD_glob_v_stdev_uniq"] = semantic_word_df['Valence'].std()
        self.features["EmoVAD_glob_a_mean_uniq"] = semantic_word_df['Arousal'].mean()
        self.features["EmoVAD_glob_a_stdev_uniq"] = semantic_word_df['Arousal'].std()
        self.features["EmoVAD_glob_d_mean_uniq"] = semantic_word_df['Dominance'].mean()
        self.features["EmoVAD_glob_d_stdev_uniq"] = semantic_word_df['Dominance'].std()

    def make_intersection(self, glob_df, wordlist_df) -> pd.DataFrame:
        semantic_word_df = pd.merge(glob_df, wordlist_df, on='Word')

        semantic_word_df['V_Total'] = semantic_word_df['Count'] * semantic_word_df['Valence']
        semantic_word_df['A_Total'] = semantic_word_df['Count'] * semantic_word_df['Arousal']
        semantic_word_df['D_Total'] = semantic_word_df['Count'] * semantic_word_df['Dominance']
        return semantic_word_df

    # For each comment in the song dataframe, clean the comment string, 
    def independent_features(self) -> None:

        columns = ['valence_means', 'valence_stds', 'valence_uniq_means', 'valence_uniq_stds',
                    'arousal_means', 'arousal_stds', 'arousal_uniq_means', 'arousal_uniq_stds',
                    'dominance_means', 'dominance_stds', 'dominance_uniq_means', 'dominance_uniq_stds']
        feature_count_df = pd.DataFrame(columns=columns)

        for i, row in enumerate(self.song_df['Comment Body']):
            # clean the string
            comment_list = clean_comment(row)
            # create a wordcount df 
            comment_df = make_word_df(comment_list)
            # Treat the comment like a tiny word glob
            semantic_word_df = self.make_intersection(comment_df, self.emovad_df)

            if semantic_word_df['Count'].sum() > 0: 
                v_mean = semantic_word_df['V_Total'].sum() / semantic_word_df['Count'].sum()
                feature_count_df.at[i,'valence_means'] = v_mean
                if(len(semantic_word_df['V_Total']) >= 2):
                    feature_count_df.at[i,'valence_stds'] = stdev(semantic_word_df['V_Total'], xbar = v_mean)
                feature_count_df.at[i,'valence_uniq_means'] = semantic_word_df['Valence'].mean()
                feature_count_df.at[i,'valence_uniq_stds'] = semantic_word_df['Valence'].mean()

                a_mean = semantic_word_df['A_Total'].sum() / semantic_word_df['Count'].sum()
                feature_count_df.at[i,'arousal_means'] = a_mean
                if(len(semantic_word_df['A_Total']) >= 2):
                    feature_count_df.at[i,'arousal_stds'] = stdev(semantic_word_df['A_Total'], xbar = a_mean)
                feature_count_df.at[i,'arousal_uniq_means'] = semantic_word_df['Arousal'].mean()
                feature_count_df.at[i,'arousal_uniq_stds'] = semantic_word_df['Arousal'].mean()

                d_mean = semantic_word_df['D_Total'].sum() / semantic_word_df['Count'].sum()
                feature_count_df.at[i,'dominance_means'] = d_mean
                if(len(semantic_word_df['D_Total']) >= 2):
                    feature_count_df.at[i,'dominance_stds'] = stdev(semantic_word_df['D_Total'], xbar = d_mean)
                feature_count_df.at[i,'dominance_uniq_means'] = semantic_word_df['Dominance'].mean()
                feature_count_df.at[i,'dominance_uniq_stds'] = semantic_word_df['Dominance'].mean()


        # Find mean of means, std of means, mean of stds, and std of stds
        
        self.features["EmoVAD_v_mean_mean"] = feature_count_df['valence_means'].mean()
        self.features["EmoVAD_v_mean_stdev"] = feature_count_df['valence_means'].std()
        self.features["EmoVAD_v_std_mean"] = feature_count_df['valence_stds'].mean()
        self.features["EmoVAD_v_stdev_stdev"] = feature_count_df['valence_stds'].std()

        self.features["EmoVAD_v_uniq_mean_mean"] = feature_count_df['valence_uniq_means'].mean()
        self.features["EmoVAD_v_uniq_mean_stdev"] = feature_count_df['valence_uniq_means'].std()
        self.features["EmoVAD_v_uniq_std_mean"] = feature_count_df['valence_uniq_stds'].mean()
        self.features["EmoVAD_v_uniq_stdev_stdev"] = feature_count_df['valence_uniq_stds'].std()

        # Arousal
        self.features["EmoVAD_a_mean_mean"] = feature_count_df['arousal_means'].mean()
        self.features["EmoVAD_a_mean_stdev"] = feature_count_df['arousal_means'].std()
        self.features["EmoVAD_a_std_mean"] = feature_count_df['arousal_stds'].mean()
        self.features["EmoVAD_a_stdev_stdev"] = feature_count_df['arousal_stds'].std()

        self.features["EmoVAD_a_uniq_mean_mean"] = feature_count_df['arousal_uniq_means'].mean()
        self.features["EmoVAD_a_uniq_mean_stdev"] = feature_count_df['arousal_uniq_means'].std()
        self.features["EmoVAD_a_uniq_std_mean"] = feature_count_df['arousal_uniq_stds'].mean()
        self.features["EmoVAD_a_uniq_stdev_stdev"] = feature_count_df['arousal_uniq_stds'].std()

        # Dominance
        self.features["EmoVAD_d_mean_mean"] = feature_count_df['dominance_means'].mean()
        self.features["EmoVAD_d_mean_stdev"] = feature_count_df['dominance_means'].std()
        self.features["EmoVAD_d_std_mean"] = feature_count_df['dominance_stds'].mean()
        self.features["EmoVAD_d_stdev_stdev"] = feature_count_df['dominance_stds'].std()

        self.features["EmoVAD_d_uniq_mean_mean"] = feature_count_df['dominance_uniq_means'].mean()
        self.features["EmoVAD_d_uniq_mean_stdev"] = feature_count_df['dominance_uniq_means'].std()
        self.features["EmoVAD_d_uniq_std_mean"] = feature_count_df['dominance_uniq_stds'].mean()
        self.features["EmoVAD_d_uniq_stdev_stdev"] = feature_count_df['dominance_uniq_stds'].std()
