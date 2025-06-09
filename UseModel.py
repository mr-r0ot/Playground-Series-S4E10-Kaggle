
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='مسیر فایل CSV ورودی')
    parser.add_argument('--models_dir', type=str, default='models/', help='مسیر مدل‌های ذخیره‌شده')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='مسیر ذخیره پیش‌بینی‌ها')
    return parser.parse_args()


def load_models(models_dir):
    models = {
        'lgb': joblib.load(os.path.join(models_dir, 'lgb_fold5.pkl')),
        'xgb': joblib.load(os.path.join(models_dir, 'xgb_fold5.pkl')),
        'cb':  joblib.load(os.path.join(models_dir, 'cb_fold5.pkl'))
    }
    return models


def feature_engineering(df_in):
    df = df_in.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in num_cols:
        df[f'{c}_2'] = df[c] ** 2
    # one-hot
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(df[cat_cols])
        df_ohe = pd.DataFrame(ohe.transform(df[cat_cols]),
                              columns=ohe.get_feature_names_out(cat_cols),
                              index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), df_ohe], axis=1)
    return df


def main():
    args = parse_args()
    df_input = pd.read_csv(args.input_file)
    if 'id' in df_input.columns:
        ids = df_input['id'].values
        df_input = df_input.drop(columns=['id'])
    else:
        ids = np.arange(len(df_input))

    X = feature_engineering(df_input)

    models = load_models(args.models_dir)

    preds_sum = np.zeros(len(X), dtype=int)
    # LightGBM
    preds_sum += (models['lgb'].predict(X) > 0.5).astype(int)
    # XGBoost
    preds_sum += (models['xgb'].predict(xgb.DMatrix(X)) > 0.5).astype(int)
    # CatBoost
    preds_sum += models['cb'].predict(X).astype(int)

    final_preds = (preds_sum >= 2).astype(int)

    df_out = pd.DataFrame({'id': ids, 'prediction': final_preds})
    df_out.to_csv(args.output_file, index=False)
    print(f"Saved {len(final_preds)} predictions to {args.output_file}")

if __name__ == '__main__':
    main()
