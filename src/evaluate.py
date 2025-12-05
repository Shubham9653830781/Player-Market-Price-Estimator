
"""Evaluate saved model on test set and print metrics and example predictions."""
import pandas as pd, argparse, joblib, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main(args):
    model = joblib.load(args.model)
    test = pd.read_csv(args.test)
    features = ['age','age_sq','goals_per_90','assists_per_90','contract_years_left','club_rank_pct','pos_bucket','nationality']
    test = test.dropna(subset=features + ['target'])
    X = test[features]
    y = test['target']
    preds = model.predict(X)
    print('RMSE (log1p):', mean_squared_error(y, preds, squared=False))
    print('MAE (log1p):', mean_absolute_error(y, preds))
    print('R2:', r2_score(y, preds))
    preds_orig = np.expm1(preds)
    y_orig = np.expm1(y)
    comp = pd.DataFrame({'pred':preds_orig[:10], 'actual': y_orig.iloc[:10]})
    print(comp.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', required=True)
    args = parser.parse_args()
    main(args)
