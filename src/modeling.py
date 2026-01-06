#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier

from .utils import get_zero_var_idx, remove_zv_by_idx

warnings.filterwarnings("ignore", message="Best weights from best epoch are automatically used!", module="pytorch_tabnet")

def oof_stacking_rf_tabnet(X_train, y_train, X_test, y_test, logger, n_splits=5, tabnet_epochs=10, tabnet_patience=5, return_fold_metrics=False):
    """
    OOF Stacking with RF + TabNet

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        logger: Logger object for logging messages
        n_splits: Number of CV folds (default: 5)
        tabnet_epochs: Max epochs for TabNet (default: 10)
        tabnet_patience: Early stopping patience for TabNet (default: 5)
        return_fold_metrics: If True, returns dict with fold-wise metrics

    Returns:
        If return_fold_metrics=False: final_test_probs (기존 동작)
        If return_fold_metrics=True: (final_test_probs, fold_metrics_dict)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_rf = np.zeros(len(X_train))
    oof_tab = np.zeros(len(X_train))
    test_preds_rf = np.zeros((len(X_test), n_splits))
    test_preds_tab = np.zeros((len(X_test), n_splits))

    # === 추가: Fold별 메트릭 저장 ===
    fold_metrics = {
        'fold': [],
        'rf_val_auc': [],
        'rf_val_acc': [],
        'tabnet_val_auc': [],
        'tabnet_val_acc': [],
        'ensemble_val_auc': [],
        'ensemble_val_acc': []
    }

    logger.log(f"Starting OOF stacking: {n_splits} folds, TabNet epochs={tabnet_epochs}, patience={tabnet_patience}")

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        logger.log(f"  Fold {fold_i+1}/{n_splits}")
        X_trn, X_val = X_train[tr_idx], X_train[va_idx]
        y_trn, y_val = y_train[tr_idx], y_train[va_idx]

        zv_idx = get_zero_var_idx(X_trn)
        X_trn_fold = remove_zv_by_idx(X_trn, zv_idx)
        X_val_fold = remove_zv_by_idx(X_val, zv_idx)
        X_test_fold = remove_zv_by_idx(X_test, zv_idx)

        # 초기화
        rf_val_auc, rf_val_acc = np.nan, np.nan
        tab_val_auc, tab_val_acc = np.nan, np.nan
        ens_val_auc, ens_val_acc = np.nan, np.nan

        if len(np.unique(y_trn)) < 2:
            logger.log(f"    Fold {fold_i+1}: Only one class in training data. Skipping model training, predicting 0.5.")
            oof_rf[va_idx] = 0.5
            oof_tab[va_idx] = 0.5
            test_preds_rf[:, fold_i] = 0.5
            test_preds_tab[:, fold_i] = 0.5
        else:
            # === RF Training ===
            try:
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_trn_fold, y_trn)
                rf_val_proba = rf.predict_proba(X_val_fold)[:, 1]
                oof_rf[va_idx] = rf_val_proba
                test_preds_rf[:, fold_i] = rf.predict_proba(X_test_fold)[:, 1]

                # === 추가: RF fold별 AUC/ACC 계산 ===
                if len(np.unique(y_val)) >= 2:
                    rf_val_auc = roc_auc_score(y_val, rf_val_proba)
                    rf_val_acc = accuracy_score(y_val, (rf_val_proba >= 0.5).astype(int))
                    logger.log(f"    Fold {fold_i+1}: RF trained. Val AUC={rf_val_auc:.4f}, Val ACC={rf_val_acc:.4f}")
                else:
                    logger.log(f"    Fold {fold_i+1}: RF trained. Val AUC=N/A (single class in val)")
            except Exception as e:
                logger.log(f"    Fold {fold_i+1}: RF training/prediction ERROR: {e}. Predicting 0.5.")
                oof_rf[va_idx] = 0.5
                test_preds_rf[:, fold_i] = 0.5

            # === TabNet Training ===
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

                    # === 추가: TabNet fold별 AUC/ACC 계산 ===
                    if len(np.unique(y_val)) >= 2:
                        tab_val_auc = roc_auc_score(y_val, pred_tab_val)
                        tab_val_acc = accuracy_score(y_val, (pred_tab_val >= 0.5).astype(int))
                        logger.log(f"    Fold {fold_i+1}: TabNet trained. Val AUC={tab_val_auc:.4f}, Val ACC={tab_val_acc:.4f}")
                    else:
                        logger.log(f"    Fold {fold_i+1}: TabNet trained. Val AUC=N/A (single class in val)")
                except RuntimeError as re:
                    logger.log(f"    Fold {fold_i+1}: TabNet training/prediction RuntimeError: {re}. Predicting 0.5.")
                except Exception as e:
                    logger.log(f"    Fold {fold_i+1}: TabNet training/prediction ERROR: {e}. Predicting 0.5.")
            else:
                logger.log(f"    Fold {fold_i+1}: Skipping TabNet training due to insufficient features ({X_trn_fold.shape[1]}). Predicting 0.5.")

            oof_tab[va_idx] = pred_tab_val
            test_preds_tab[:, fold_i] = pred_tab_test

            # === 추가: Ensemble (RF + TabNet 평균) fold별 AUC/ACC ===
            ensemble_val_proba = (oof_rf[va_idx] + oof_tab[va_idx]) / 2
            try:
                if len(np.unique(y_val)) >= 2:
                    ens_val_auc = roc_auc_score(y_val, ensemble_val_proba)
                    ens_val_acc = accuracy_score(y_val, (ensemble_val_proba >= 0.5).astype(int))
                    logger.log(f"    Fold {fold_i+1}: Ensemble Val AUC={ens_val_auc:.4f}, Val ACC={ens_val_acc:.4f}")
            except Exception as e:
                logger.log(f"    Fold {fold_i+1}: Ensemble metric calculation error: {e}")

        # === Fold 메트릭 저장 ===
        fold_metrics['fold'].append(fold_i + 1)
        fold_metrics['rf_val_auc'].append(rf_val_auc)
        fold_metrics['rf_val_acc'].append(rf_val_acc)
        fold_metrics['tabnet_val_auc'].append(tab_val_auc)
        fold_metrics['tabnet_val_acc'].append(tab_val_acc)
        fold_metrics['ensemble_val_auc'].append(ens_val_auc)
        fold_metrics['ensemble_val_acc'].append(ens_val_acc)

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

    # === 추가: Summary 로깅 ===
    logger.log("=" * 50)
    logger.log("Fold-wise Metrics Summary:")
    for i in range(n_splits):
        rf_auc_str = f"{fold_metrics['rf_val_auc'][i]:.4f}" if not np.isnan(fold_metrics['rf_val_auc'][i]) else "N/A"
        tab_auc_str = f"{fold_metrics['tabnet_val_auc'][i]:.4f}" if not np.isnan(fold_metrics['tabnet_val_auc'][i]) else "N/A"
        ens_auc_str = f"{fold_metrics['ensemble_val_auc'][i]:.4f}" if not np.isnan(fold_metrics['ensemble_val_auc'][i]) else "N/A"
        logger.log(f"  Fold {i+1}: RF AUC={rf_auc_str}, TabNet AUC={tab_auc_str}, Ensemble AUC={ens_auc_str}")

    # Mean ± Std 계산
    rf_aucs = [x for x in fold_metrics['rf_val_auc'] if not np.isnan(x)]
    tab_aucs = [x for x in fold_metrics['tabnet_val_auc'] if not np.isnan(x)]
    ens_aucs = [x for x in fold_metrics['ensemble_val_auc'] if not np.isnan(x)]

    if rf_aucs:
        logger.log(f"RF Mean AUC: {np.mean(rf_aucs):.4f} +/- {np.std(rf_aucs):.4f}")
    if tab_aucs:
        logger.log(f"TabNet Mean AUC: {np.mean(tab_aucs):.4f} +/- {np.std(tab_aucs):.4f}")
    if ens_aucs:
        logger.log(f"Ensemble Mean AUC: {np.mean(ens_aucs):.4f} +/- {np.std(ens_aucs):.4f}")
    logger.log("=" * 50)

    if return_fold_metrics:
        return final_test_probs, fold_metrics
    return final_test_probs
