import pandas as pd

class EmoVADGenerator:

    features = {
        "EmoVAD_glob_v_mean": 0.0, "EmoVAD_glob_v_stdev": 0.0,
        "EmoVAD_glob_a_mean": 0.0, "EmoVAD_glob_a_stdev": 0.0,
        "EmoVAD_glob_d_mean": 0.0, "EmoVAD_glob_d_stdev": 0.0,
        "EmoVAD_glob_v_mean_uniq": 0.0, "EmoVAD_glob_v_stdev_uniq": 0.0,
        "EmoVAD_glob_a_mean_uniq": 0.0, "EmoVAD_glob_a_stdev_uniq": 0.0,
        "EmoVAD_glob_d_mean_uniq": 0.0, "EmoVAD_glob_d_stdev_uniq": 0.0
    }

    def __init__(self, song_df, glob_df, emovad_df) -> None:
        self.song_df = song_df
        self.glob_df = glob_df
        self.emovad_df = emovad_df

    def get_features(self) -> dict:
        self.glob_features()
        return self.features

    
    # Find the features of the word glob for a specific song 
    def glob_features(self) -> None:
        semantic_word_df = pd.merge(self.glob_df, self.emovad_df, on='Word')

        semantic_word_df['V_Total'] = semantic_word_df['Count'] * self.emovad_df['Valence']
        semantic_word_df['A_Total'] = semantic_word_df['Count'] * self.emovad_df['Arousal']
        semantic_word_df['D_Total'] = semantic_word_df['Count'] * self.emovad_df['Dominance']
        #VAD means including duplicates
        self.features["EmoVAD_glob_v_mean"] = semantic_word_df['V_Total'].mean()
        self.features["EmoVAD_glob_v_stdev"] = semantic_word_df['V_Total'].std()
        self.features["EmoVAD_glob_a_mean"] = semantic_word_df['A_Total'].mean()
        self.features["EmoVAD_glob_a_stdev"] = semantic_word_df['A_Total'].std()
        self.features["EmoVAD_glob_d_mean"] = semantic_word_df['D_Total'].mean()
        self.features["EmoVAD_glob_d_stdev"] = semantic_word_df['D_Total'].std()

        #VAD means on unique words only
        self.features["EmoVAD_glob_v_mean_uniq"] = semantic_word_df['Valence'].mean()
        self.features["EmoVAD_glob_v_stdev_uniq"] = semantic_word_df['Valence'].std()
        self.features["EmoVAD_glob_a_mean_uniq"] = semantic_word_df['Arousal'].mean()
        self.features["EmoVAD_glob_a_stdev_uniq"] = semantic_word_df['Arousal'].std()
        self.features["EmoVAD_glob_d_mean_uniq"] = semantic_word_df['Dominance'].mean()
        self.features["EmoVAD_glob_d_stdev_uniq"] = semantic_word_df['Dominance'].std()
