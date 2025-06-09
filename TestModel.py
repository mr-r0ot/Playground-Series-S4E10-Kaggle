# -*- coding: utf-8 -*-
"""
TestModel.py

این اسکریپت برای:
 - بارگذاری مدل‌های ذخیره‌شده فولد ۵ (LightGBM, XGBoost, CatBoost)
 - بارگذاری دیتاست train برای ارزیابی OOF و دیتاست test برای سرعت inference
 - محاسبه معیارهای OOF (Accuracy, Precision, Recall, F1)
 - زمان‌سنجی inference روی تست
 - رسم و ذخیره نمودارها و گزارش‌ها در پوشه‌ی TestModel/

Usage:
    python TestModel.py --data_dir data/ --models_dir models/ --oof_path models/oof.csv --output_dir TestModel/

نیازمندی‌ها:
    pandas, numpy, joblib, lightgbm, xgboost, catboost, sklearn, matplotlib
"""
import os
import argparse
import time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='مسیر داده‌ها')
    parser.add_argument('--models_dir', type=str, default='models/', help='مسیر مدل‌های ذخیره‌شده')
    parser.add_argument('--oof_path', type=str, default='models/oof.csv', help='مسیر فایل OOF پیش‌بینی‌ها')
    parser.add_argument('--output_dir', type=str, default='TestModel/', help='مسیر ذخیره نتایج و نمودارها')
    return parser.parse_args()


def load_data_and_oof(data_dir, oof_path):
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_oof = pd.read_csv(oof_path)
    merge = df_train.merge(df_oof, on='id')
    X = merge.drop(columns=['loan_status', 'oof_pred'])
    y_true = merge['loan_status'].values
    y_pred_oof = merge['oof_pred'].values
    return X, y_true, y_pred_oof


def load_test_features(data_dir):
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    ids = df_test['id'].values
    X = df_test.drop(columns=['id'])
    return X, ids


def feature_engineering(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in num_cols:
        X[f'{c}_2'] = X[c]**2
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X[cat_cols])
    df_ohe = pd.DataFrame(ohe.transform(X[cat_cols]), columns=ohe.get_feature_names_out(), index=X.index)
    X_final = pd.concat([X.drop(columns=cat_cols), df_ohe], axis=1)
    return X_final


def load_models(models_dir):
    # بارگذاری مدل‌های فولد 5 با نام صحیح
    models = {
        'LightGBM': joblib.load(os.path.join(models_dir, 'lgb_fold5.pkl')),
        'XGBoost': joblib.load(os.path.join(models_dir, 'xgb_fold5.pkl')),
        'CatBoost': joblib.load(os.path.join(models_dir, 'cb_fold5.pkl'))
    }
    return models


def evaluate_oof(y_true, y_pred, output_dir):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    df = pd.DataFrame([{'Model':'OOF Ensemble', 'Accuracy':acc, 'Precision':prec, 'Recall':rec, 'F1':f1}])
    df.to_csv(os.path.join(output_dir, 'oof_metrics.csv'), index=False)
    return df


def evaluate_speed(models, X_test, output_dir):
    metrics = []
    for name, model in models.items():
        start = time.time()
        if name == 'XGBoost':
            _ = model.predict(xgb.DMatrix(X_test))
        else:
            _ = model.predict(X_test)
        elapsed = (time.time() - start) * 1000
        metrics.append({'Model':name, 'Inference Time (ms)':elapsed})
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, 'inference_time.csv'), index=False)
    return df


def plot_and_save(df, x, y, title, ylabel, fname, output_dir):
    plt.figure()
    plt.bar(df[x], df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X_oof, y_true, y_pred_oof = load_data_and_oof(args.data_dir, args.oof_path)
    df_oof = evaluate_oof(y_true, y_pred_oof, args.output_dir)

    X_test, ids_test = load_test_features(args.data_dir)
    X_test_fe = feature_engineering(X_test.copy())
    models = load_models(args.models_dir)
    df_time = evaluate_speed(models, X_test_fe, args.output_dir)

    plot_and_save(df_oof, 'Model', 'Accuracy', 'OOF Accuracy', 'Accuracy', 'oof_accuracy.png', args.output_dir)
    plot_and_save(df_time, 'Model', 'Inference Time (ms)', 'Inference Time by Model', 'Time (ms)', 'inference_time.png', args.output_dir)

    print('OOF Metrics:')
    print(df_oof)
    print('Inference Times:')
    print(df_time)

if __name__ == '__main__':
    main()
