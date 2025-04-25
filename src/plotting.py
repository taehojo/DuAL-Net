#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def evaluate_snp_subset(raw_df, snps, y, n_splits=5, logger=None):
    if not snps:
        if logger: logger.log("    evaluate_snp_subset: SNP list is empty. Returning accuracy 0.0.")
        return 0.0
    if not all(snp in raw_df.columns for snp in snps):
        if logger: logger.log("    evaluate_snp_subset: Some requested SNPs not found in raw_df. Proceeding with available ones.")
        snps = [snp for snp in snps if snp in raw_df.columns]
        if not snps:
             if logger: logger.log("    evaluate_snp_subset: No valid SNPs found after filtering. Returning accuracy 0.0.")
             return 0.0

    X = raw_df[snps].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if len(np.unique(y_tr)) < 2:
            if logger: logger.log(f"      evaluate_snp_subset Fold {fold+1}: Only one class in training split. Skipping fold.")
            continue

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        try:
            rf.fit(X_tr_scaled, y_tr)
            preds = rf.predict(X_val_scaled)
            acc = accuracy_score(y_val, preds)
            accs.append(acc)
        except Exception as e:
             if logger: logger.log(f"      evaluate_snp_subset Fold {fold+1}: Error during RF training/prediction: {e}. Skipping fold.")
             continue

    if not accs:
        if logger: logger.log("    evaluate_snp_subset: No folds completed successfully. Returning accuracy 0.0.")
        return 0.0

    return np.mean(accs)

def get_roc_data(raw_df, snps, y, n_splits=5, logger=None):
    if not snps:
        if logger: logger.log("    get_roc_data: SNP list is empty. Returning default ROC data (0 AUC).")
        return np.array([0, 1]), np.array([0, 1]), 0.0
    if not all(snp in raw_df.columns for snp in snps):
        if logger: logger.log("    get_roc_data: Some requested SNPs not found in raw_df. Proceeding with available ones.")
        snps = [snp for snp in snps if snp in raw_df.columns]
        if not snps:
            if logger: logger.log("    get_roc_data: No valid SNPs found after filtering. Returning default ROC data (0 AUC).")
            return np.array([0, 1]), np.array([0, 1]), 0.0

    X = raw_df[snps].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_all_folds = []
    prob_all_folds = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if len(np.unique(y_tr)) < 2:
             if logger: logger.log(f"      get_roc_data Fold {fold+1}: Only one class in training split. Skipping fold.")
             continue

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        try:
            rf.fit(X_tr_scaled, y_tr)
            probs = rf.predict_proba(X_val_scaled)[:, 1]
            y_all_folds.append(y_val)
            prob_all_folds.append(probs)
        except Exception as e:
            if logger: logger.log(f"      get_roc_data Fold {fold+1}: Error during RF training/prediction: {e}. Skipping fold.")
            continue

    if not y_all_folds:
        if logger: logger.log("    get_roc_data: No folds completed successfully. Returning default ROC data (0 AUC).")
        return np.array([0, 1]), np.array([0, 1]), 0.0

    y_true_all = np.concatenate(y_all_folds)
    y_prob_all = np.concatenate(prob_all_folds)

    if len(np.unique(y_true_all)) < 2:
         if logger: logger.log("    get_roc_data: Only one class present in combined validation sets. Cannot compute AUC. Returning default.")
         return np.array([0, 1]), np.array([0, 1]), 0.0

    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def plot_roc_for_n(n, ranked_df, raw_df, y, pdf_object, logger, n_splits=5):
    logger.log(f"Generating ROC curves for N = {n}")
    num_available_snps = len(ranked_df)

    if num_available_snps == 0:
        logger.log(f"  No SNPs available in ranked_df. Skipping ROC plot for N={n}.")
        return

    n_eff = min(n, num_available_snps)

    topN = ranked_df.head(n_eff)['SNP'].tolist()
    bottomN = ranked_df.tail(n_eff)['SNP'].tolist()
    randomN = ranked_df['SNP'].sample(n=n_eff, random_state=42).tolist()

    logger.log(f"  Calculating ROC for Top {n_eff}...")
    fpr_top, tpr_top, auc_top = get_roc_data(raw_df, topN, y, n_splits, logger)
    logger.log(f"  Calculating ROC for Bottom {n_eff}...")
    fpr_bot, tpr_bot, auc_bot = get_roc_data(raw_df, bottomN, y, n_splits, logger)
    logger.log(f"  Calculating ROC for Random {n_eff}...")
    fpr_rand, tpr_rand, auc_rand = get_roc_data(raw_df, randomN, y, n_splits, logger)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_top, tpr_top, lw=2, label=f'Top {n_eff} SNPs (AUC = {auc_top:.3f})')
    plt.plot(fpr_bot, tpr_bot, lw=2, label=f'Bottom {n_eff} SNPs (AUC = {auc_bot:.3f})')
    plt.plot(fpr_rand, tpr_rand, lw=2, label=f'Random {n_eff} SNPs (AUC = {auc_rand:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curves for SNP Subsets (N = {n_eff})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    pdf_object.savefig()
    plt.close()
    logger.log(f"  ROC plot for N = {n_eff} added to PDF.")
