
"""Preprocessing for Player Market Price Estimator
Converts currency strings, imputes missing, creates per90 features and splits train/test.
"""
import pandas as pd, numpy as np, argparse, re
from sklearn.model_selection import train_test_split

def money_to_num(s):
    if pd.isna(s): return np.nan
    s = str(s).replace('\u20ac','').replace('€','').strip()
    s = s.replace(',','')
    if s == '': return np.nan
    try:
        if s.endswith('M'): return float(s[:-1]) * 1e6
        if s.endswith('K'): return float(s[:-1]) * 1e3
        return float(s)
    except:
        return np.nan

def preprocess(df):
    df = df.copy()
    df['market_value_num'] = df['market_value'].apply(money_to_num)
    df = df.dropna(subset=['market_value_num'])
    # filter unrealistic ages
    df = df[(df['age']>=15)&(df['age']<=50)]
    # per90 features
    df['minutes_played'] = df['minutes_played'].fillna(0)
    df['goals_per_90'] = df['goals'] / (df['minutes_played']/90).replace([np.inf,-np.inf], np.nan)
    df['assists_per_90'] = df['assists'] / (df['minutes_played']/90).replace([np.inf,-np.inf], np.nan)
    df['goals_per_90'] = df['goals_per_90'].fillna(0)
    df['assists_per_90'] = df['assists_per_90'].fillna(0)
    # log1p target to reduce skew
    df['target'] = np.log1p(df['market_value_num'])
    return df

def main(args):
    df = pd.read_csv(args.input)
    df = preprocess(df)
    # add a simple position bucket
    mapping = {'ST':'Forward','LW':'Forward','RW':'Forward','CF':'Forward',
               'CM':'Midfielder','CDM':'Midfielder','CAM':'Midfielder',
               'CB':'Defender','LB':'Defender','RB':'Defender','GK':'Goalkeeper'}
    df['pos_bucket'] = df['position'].map(mapping).fillna('Other')
    # add simple club reputation
    grp = df.groupby('club')['market_value_num'].median().rename('club_med_mv')
    df = df.merge(grp, on='club', how='left')
    df['club_rank_pct'] = df['club_med_mv'].rank(pct=True)
    # split and save
    train, test = train_test_split(df, test_size=0.15, random_state=42)
    train.to_csv(args.output.replace('.csv','_train.csv'), index=False)
    test.to_csv(args.output.replace('.csv','_test.csv'), index=False)
    df.to_csv(args.output, index=False)
    print('Saved processed files. Train size:', len(train), 'Test size:', len(test))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args)
