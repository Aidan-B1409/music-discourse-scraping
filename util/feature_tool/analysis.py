from numpy.core.fromnumeric import var
import pandas as pd
import csv

meta_cols = 10
def analyze_features(feature_fname, analysis_fname, feature_keys):
    data = pd.read_csv(feature_fname, encoding="utf-8", index_col = False)
    analysis_df = pd.DataFrame(columns=feature_keys.keys())

    variance = data.var(skipna=True, numeric_only=True)
    variance.name = 'variance'
    analysis_df = analysis_df.append(variance)

    # valence_correl = [data['existing_valence'].corr(data[col], method='spearman') for col in data[meta_cols:]]
    valence_correl = data.corrwith(data['existing_valence'], method='spearman')
    valence_correl = pd.Series(valence_correl, name='valence_correl_spearman')
    analysis_df = analysis_df.append(valence_correl)

    valence_correl = data.corrwith(data['existing_valence'], method='kendall')
    valence_correl = pd.Series(valence_correl, name='valence_correl_kendall')
    analysis_df = analysis_df.append(valence_correl)

    valence_correl = data.corrwith(data['existing_valence'], method='pearson')
    valence_correl = pd.Series(valence_correl, name='valence_correl_pearson')
    analysis_df = analysis_df.append(valence_correl)

    arousal_correl = data.corrwith(data['existing_arousal'], method='spearman')
    arousal_correl = pd.Series(arousal_correl, name='arousal_correl_spearman')
    analysis_df = analysis_df.append(arousal_correl)

    arousal_correl = data.corrwith(data['existing_arousal'], method='kendall')
    arousal_correl = pd.Series(arousal_correl, name='arousal_correl_kendall')
    analysis_df = analysis_df.append(arousal_correl)

    arousal_correl = data.corrwith(data['existing_arousal'], method='pearson')
    arousal_correl = pd.Series(arousal_correl, name='arousal_correl_pearson')
    analysis_df = analysis_df.append(arousal_correl)

    analysis_df.to_csv(analysis_fname)