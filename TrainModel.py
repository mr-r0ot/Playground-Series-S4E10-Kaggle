
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='مسیر داده‌ها')
    parser.add_argument('--output_dir', type=str, default='models/', help='مسیر خروجی مدل‌ها و نتایج')
    parser.add_argument('--n_splits', type=int, default=5, help='تعداد فولد CV')
    return parser.parse_args()


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("train.csv یا test.csv یافت نشد.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print("Columns in train:", train.columns.tolist())
    print("Columns in test :", test.columns.tolist())

    train_ids = train['id'].values
    test_ids = test['id'].values

    if pd.api.types.is_numeric_dtype(train['loan_status']):
        y = train['loan_status'].values
    else:
        mapping = {"Fully Paid": 0, "Charged Off": 1}
        y = train['loan_status'].map(mapping).values

    X_train = train.drop(columns=['id', 'loan_status'])
    X_test = test.drop(columns=['id'])

    return X_train, y, train_ids, X_test, test_ids


def feature_engineering(X_train, X_test):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    for c in num_cols:
        X_train[f'{c}_2'] = X_train[c] ** 2
        X_test[f'{c}_2'] = X_test[c] ** 2

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[cat_cols])
    train_ohe = pd.DataFrame(encoder.transform(X_train[cat_cols]),
                             columns=encoder.get_feature_names_out(cat_cols),
                             index=X_train.index)
    test_ohe = pd.DataFrame(encoder.transform(X_test[cat_cols]),
                            columns=encoder.get_feature_names_out(cat_cols),
                            index=X_test.index)

    X_train_final = pd.concat([X_train[num_cols + [f'{c}_2' for c in num_cols]].reset_index(drop=True),
                                train_ohe.reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test[num_cols + [f'{c}_2' for c in num_cols]].reset_index(drop=True),
                               test_ohe.reset_index(drop=True)], axis=1)
    return X_train_final, X_test_final


def train_and_predict(X, y, train_ids, X_test, test_ids, args):
    oof_preds = np.zeros(len(y), dtype=int)
    test_preds_sum = np.zeros(len(test_ids), dtype=int)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"=== Fold {fold}/{args.n_splits} ===")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # LightGBM
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        lgb_params = {'objective':'binary','metric':'binary_error','learning_rate':0.05,
                      'num_leaves':64,'feature_fraction':0.8,'bagging_fraction':0.8,
                      'bagging_freq':1,'seed':42,'verbosity':-1}
        model_lgb = lgb.train(
            lgb_params, lgb_train, num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )

        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_params = {'objective':'binary:logistic','eval_metric':'error','learning_rate':0.05,
                      'max_depth':7,'subsample':0.8,'colsample_bytree':0.8,'seed':42}
        model_xgb = xgb.train(
            xgb_params, dtrain, num_boost_round=model_lgb.best_iteration,
            evals=[(dtrain,'train'),(dval,'valid')], early_stopping_rounds=50,
            verbose_eval=False
        )

        # CatBoost
        model_cb = CatBoostClassifier(
            iterations=model_lgb.best_iteration, learning_rate=0.05, depth=6,
            eval_metric='Accuracy', random_seed=42, verbose=False
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_val,y_val), use_best_model=True, verbose=False)

        p_lgb = (model_lgb.predict(X_val) > 0.5).astype(int)
        p_xgb = (model_xgb.predict(xgb.DMatrix(X_val)) > 0.5).astype(int)
        p_cb = model_cb.predict(X_val).astype(int)
        oof_preds[val_idx] = ((p_lgb + p_xgb + p_cb) >= 2).astype(int)

        t_lgb = (model_lgb.predict(X_test) > 0.5).astype(int)
        t_xgb = (model_xgb.predict(xgb.DMatrix(X_test)) > 0.5).astype(int)
        t_cb = model_cb.predict(X_test).astype(int)
        test_preds_sum += ((t_lgb + t_xgb + t_cb) >= 2).astype(int)

        os.makedirs(args.output_dir, exist_ok=True)
        joblib.dump(model_lgb, os.path.join(args.output_dir, f'lgb_fold{fold}.pkl'))
        joblib.dump(model_xgb, os.path.join(args.output_dir, f'xgb_fold{fold}.pkl'))
        joblib.dump(model_cb, os.path.join(args.output_dir, f'cb_fold{fold}.pkl'))

    final_test = (test_preds_sum / args.n_splits) >= 0.5
    oof_acc = accuracy_score(y, oof_preds)
    print(f"OOF Accuracy: {oof_acc:.4f}")
    if oof_acc < 0.95:
        print("Warning: OOF accuracy is onder the 95%!")

    pd.DataFrame({'id':train_ids,'oof_pred':oof_preds}).to_csv(
        os.path.join(args.output_dir,'oof.csv'), index=False)
    pd.DataFrame({'id':test_ids,'loan_status':final_test.astype(int)}).to_csv(
        os.path.join(args.output_dir,'submission.csv'), index=False)


if __name__ == '__main__':
    args = parse_args()
    X_train, y, train_ids, X_test, test_ids = load_data(args.data_dir)
    X_train_fe, X_test_fe = feature_engineering(X_train, X_test)
    train_and_predict(X_train_fe, y, train_ids, X_test_fe, test_ids, args)