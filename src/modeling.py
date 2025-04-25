#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier

from .utils import get_zero_var_idx, remove_zv_by_idx

warnings.filterwarnings("ignore", message="Best weights from best epoch are automatically used!", module="pytorch_tabnet")

def oof_stacking_rf_tabnet(X_train, y_train, X_test, y_test, logger, n_splits=5, tabnet_epochs=10, tabnet_patience=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_rf = np.zeros(len(X_train))
    oof_tab = np.zeros(len(X_train))
    test_preds_rf = np.zeros((len(X_test), n_splits))
    test_preds_tab = np.zeros((len(X_test), n_splits))

    logger.log(f"Starting OOF stacking: {n_splits} folds, TabNet epochs={tabnet_epochs}, patience={tabnet_patience}")

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        logger.log(f"  Fold {fold_i+1}/{n_splits}")
        X_trn, X_val = X_train[tr_idx], X_train[va_idx]
        y_trn, y_val = y_train[tr_idx], y_train[va_idx]

        zv_idx = get_zero_var_idx(X_trn)
        X_trn_fold = remove_zv_by_idx(X_trn, zv_idx)
        X_val_fold = remove_zv_by_idx(X_val, zv_idx)
        X_test_fold = remove_zv_by_idx(X_test, zv_idx)

        if len(np.unique(y_trn)) < 2:
            logger.log(f"    Fold {fold_i+1}: Only one class in training data. Skipping model training, predicting 0.5.")
            oof_rf[va_idx] = 0.5
            oof_tab[va_idx] = 0.5
            test_preds_rf[:, fold_i] = 0.5
            test_preds_tab[:, fold_i] = 0.5
            continue

        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_trn_fold, y_trn)
            oof_rf[va_idx] = rf.predict_proba(X_val_fold)[:, 1]
            test_preds_rf[:, fold_i] = rf.predict_proba(X_test_fold)[:, 1]
            logger.log(f"    Fold {fold_i+1}: RF trained.")
        except Exception as e:
            logger.log(f"    Fold {fold_i+1}: RF training/prediction ERROR: {e}. Predicting 0.5.")
            oof_rf[va_idx] = 0.5
            test_preds_rf[:, fold_i] = 0.5

        pred_tab_val = np.full(len(X_val_fold), 0.5)
        pred_tab_test = np.full(len(X_test_fold), 0.5)

        if X_trn_fold.shape[1] >= 1:
            tab = TabNetClassifier(
                n_d=8, n_a=8, n_steps=3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=1e-3),
                scheduler_params={"step_size": 50, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type="entmax",
                device_name='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=0
            )
            try:
                tab.fit(
                    X_train=X_trn_fold, y_train=y_trn,
                    eval_set=[(X_val_fold, y_val)],
                    max_epochs=tabnet_epochs,
                    patience=tabnet_patience,
                    eval_metric=["auc"],
                    batch_size=1024,
                    virtual_batch_size=128,
                    drop_last=False
                )
                pred_tab_val = tab.predict_proba(X_val_fold)[:, 1]
                pred_tab_test = tab.predict_proba(X_test_fold)[:, 1]
                logger.log(f"    Fold {fold_i+1}: TabNet trained.")
            except RuntimeError as re:
                logger.log(f"    Fold {fold_i+1}: TabNet training/prediction RuntimeError: {re}. Predicting 0.5.")
            except Exception as e:
                 logger.log(f"    Fold {fold_i+1}: TabNet training/prediction ERROR: {e}. Predicting 0.5.")
        else:
            logger.log(f"    Fold {fold_i+1}: Skipping TabNet training due to insufficient features ({X_trn_fold.shape[1]}). Predicting 0.5.")

        oof_tab[va_idx] = pred_tab_val
        test_preds_tab[:, fold_i] = pred_tab_test

    test_pred_rf_mean = np.mean(test_preds_rf, axis=1)
    test_pred_tab_mean = np.mean(test_preds_tab, axis=1)

    logger.log("Training meta-learner (Logistic Regression)...")
    meta_model = LogisticRegression(random_state=42)
    X_meta_train = np.column_stack([oof_rf, oof_tab])
    meta_model.fit(X_meta_train, y_train)
    logger.log("Meta-learner trained.")

    X_meta_test = np.column_stack([test_pred_rf_mean, test_pred_tab_mean])
    final_test_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    logger.log("Generated final predictions on the test set.")

    return final_test_probs
