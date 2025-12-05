
"""Train and tune models using GridSearchCV and save best pipeline."""
import pandas as pd, argparse, joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from pipeline_builder import build_preprocessor

def main(args):
    df = pd.read_csv(args.input)
    # add engineered features if not present
    if 'age_sq' not in df.columns:
        df['age_sq'] = df['age'] ** 2
    features = ['age','age_sq','goals_per_90','assists_per_90','contract_years_left','club_rank_pct']
    cat = ['pos_bucket','nationality']
    df = df.dropna(subset=features + ['target'])
    X = df[features + cat]
    y = df['target']
    preproc = build_preprocessor(num_features=features, cat_features=cat, high_card_cats=['nationality'])
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('preproc', preproc), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])
    param_grid = {
        'model__n_estimators':[100,200],
        'model__max_depth':[8,12,None],
        'model__min_samples_split':[2,5]
    }
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=1, verbose=2)
    gs.fit(X, y)
    print('Best params:', gs.best_params_)
    best = gs.best_estimator_
    joblib.dump(best, args.out_dir.rstrip('/') + '/best_model.pkl')
    print('Saved best model. Train RMSE:', mean_squared_error(y, best.predict(X), squared=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    main(args)
