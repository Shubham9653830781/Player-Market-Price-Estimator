
"""Feature engineering helpers."""
import pandas as pd, numpy as np

def add_age_features(df):
    df = df.copy()
    df['age_sq'] = df['age'] ** 2
    df['is_young'] = (df['age'] < 23).astype(int)
    return df

def club_reputation(df):
    grp = df.groupby('club')['market_value_num'].median().rename('club_med_mv')
    df = df.merge(grp, on='club', how='left')
    df['club_rank_pct'] = df['club_med_mv'].rank(pct=True)
    return df
