#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

from .utils import Logger, split_into_windows_as_sequence, get_zero_var_idx, remove_zv_by_idx
from .modeling import oof_stacking_rf_tabnet

def local_analysis(raw_df, y, window_size, prefix, tabnet_epoch, tabnet_patience, n_splits, logger, result_dir="result"):
    X_seq = split_into_windows_as_sequence(raw_df, window_size)
    if X_seq.size == 0:
         logger.log("Local Analysis: No windows were created (possibly due to window_size > num_features). Skipping local analysis.")
         return pd.DataFrame({'SNP': [], 'window_id': [], 'local_accuracy': []})

    N, W, dim = X_seq.shape
    logger.log(f"Starting Local Analysis: N={N}, W={W} windows, window_size={dim}")

    local_accuracies = np.zeros(W)
    acc_dict = {}
    wdict = {}

    X_placeholder = np.zeros((N, 1))
    _, _, y_train_outer, y_test_outer = train_test_split(X_placeholder, y, test_size=0.2, random_state=42, stratify=y)

    for w in range(W):
        logger.log(f"  Processing Window {w+1}/{W}")
        X_window = X_seq[:, w]

        X_win_train, X_win_test, y_win_train, y_win_test = train_test_split(
            X_window, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_win_train_scaled = scaler.fit_transform(X_win_train)
        X_win_test_scaled = scaler.transform(X_win_test)

        if X_win_train_scaled.shape[0] < n_splits * 2 or len(np.unique(y_win_train)) < 2:
             logger.log(f"    Window {w+1}: Insufficient data or only one class in training split. Accuracy set to 0.5.")
             acc_win = 0.5
        else:
            test_probs = oof_stacking_rf_tabnet(
                X_win_train_scaled, y_win_train,
                X_win_test_scaled, y_win_test,
                logger=logger,
                n_splits=n_splits,
                tabnet_epochs=tabnet_epoch,
                tabnet_patience=tabnet_patience
            )
            test_preds = (test_probs >= 0.5).astype(int)
            acc_win = np.mean(test_preds == y_win_test)

        logger.log(f"    Window {w+1} accuracy = {acc_win:.4f}")
        local_accuracies[w] = acc_win

        st = w * window_size
        en = st + window_size
        snps_in_window = raw_df.columns[st:en]
        for snp in snps_in_window:
            acc_dict[snp] = acc_win
            wdict[snp] = w + 1

    avg_local = np.mean(local_accuracies) if W > 0 else 0.0
    logger.log(f"Overall average local accuracy: {avg_local:.4f}")

    df_local = pd.DataFrame({
        'SNP': list(acc_dict.keys()),
        'window_id': [wdict[s] for s in acc_dict.keys()],
        'local_accuracy': [acc_dict[s] for s in acc_dict.keys()]
    })

    os.makedirs(result_dir, exist_ok=True)
    out_file = os.path.join(result_dir, f"{prefix}_local_results.csv")
    df_local.to_csv(out_file, index=False)
    logger.log(f"Local analysis results saved -> {out_file}")
    return df_local

def calculate_global_annotation(raw_df, y, anno_df, prefix, tabnet_epoch, tabnet_patience, n_splits, logger, result_dir="result"):
    if 'rs_id' not in anno_df.columns:
        logger.log("Error: 'rs_id' column missing in annotation DataFrame! Skipping global analysis.")
        return pd.DataFrame()

    if anno_df.index.name != 'rs_id':
        if 'rs_id' in anno_df.columns:
            if anno_df['rs_id'].duplicated().any():
                logger.log("Warning: Duplicate rs_ids found in annotation data. Aggregating features.")
                anno_df = anno_df.drop_duplicates(subset=['rs_id'], keep='first')
            anno_df = anno_df.set_index('rs_id')
        else:
             logger.log("Error: Cannot set 'rs_id' as index. Column missing.")
             return pd.DataFrame()

    potential_anno_cols = [c for c in anno_df.columns if is_numeric_dtype(anno_df[c])]
    logger.log(f"Found {len(potential_anno_cols)} potential numeric annotation columns.")

    results_list = []

    for col in potential_anno_cols:
        logger.log(f"  [Global] Processing annotation: {col}")

        subset_snps_anno = anno_df.index[anno_df[col] > 0].tolist()

        if not subset_snps_anno:
            logger.log(f"    No SNPs found with {col} > 0. Skipping.")
            continue

        relevant_snps_in_raw = [snp for snp in subset_snps_anno if snp in raw_df.columns]

        if not relevant_snps_in_raw:
            logger.log(f"    None of the SNPs for annotation '{col}' are present in the raw data. Skipping.")
            continue

        logger.log(f"    Found {len(relevant_snps_in_raw)} relevant SNPs in raw data for '{col}'.")

        X_sub = raw_df[relevant_snps_in_raw].values

        X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(
            X_sub, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_sub_train_scaled = scaler.fit_transform(X_sub_train)
        X_sub_test_scaled = scaler.transform(X_sub_test)

        acc_global = 0.5
        if X_sub_train_scaled.shape[0] < n_splits * 2 or len(np.unique(y_sub_train)) < 2:
             logger.log(f"    Annotation '{col}': Insufficient data or only one class in training split. Accuracy set to 0.5.")
        else:
            test_probs = oof_stacking_rf_tabnet(
                X_sub_train_scaled, y_sub_train,
                X_sub_test_scaled, y_sub_test,
                logger=logger,
                n_splits=n_splits,
                tabnet_epochs=tabnet_epoch,
                tabnet_patience=tabnet_patience
            )
            test_preds = (test_probs >= 0.5).astype(int)
            acc_global = np.mean(test_preds == y_sub_test)

        logger.log(f"    Global accuracy for annotation '{col}': {acc_global:.4f}")

        accuracy_col_name = f"{col}_global_accuracy"
        for snp in relevant_snps_in_raw:
            results_list.append({'SNP': snp, accuracy_col_name: acc_global})

    if not results_list:
        logger.log("Global Analysis: No results generated.")
        return pd.DataFrame({'SNP': []})

    df_results_global = pd.DataFrame(results_list)
    df_global_agg = df_results_global.groupby('SNP').first().reset_index()

    os.makedirs(result_dir, exist_ok=True)
    out_file = os.path.join(result_dir, f"{prefix}_global_results.csv")
    df_global_agg.to_csv(out_file, index=False)
    logger.log(f"Global analysis results saved -> {out_file}")

    return df_global_agg

def alpha_search(merged_df, raw_df, y, logger, top_n=100, n_splits=5, step=0.1):
    logger.log(f"Starting Alpha Search (Top {top_n} SNPs, step={step})...")
    alpha_list = []
    score_list = []
    best_alpha = 0.0
    best_score = -1.0

    if 'local_accuracy' not in merged_df.columns:
        logger.log("Warning: 'local_accuracy' not found in merged_df. Skipping alpha search related to local score.")
        return pd.DataFrame({'alpha': [], 'CV_accuracy': []}), 0.0, 0.0

    if 'global_accuracy_mean' not in merged_df.columns:
         logger.log("Warning: 'global_accuracy_mean' not found in merged_df. Assuming 0 for global score in alpha search.")
         merged_df['global_accuracy_mean'] = 0.0


    merged_df['local_accuracy'] = pd.to_numeric(merged_df['local_accuracy'], errors='coerce').fillna(0.0)
    merged_df['global_accuracy_mean'] = pd.to_numeric(merged_df['global_accuracy_mean'], errors='coerce').fillna(0.0)

    for alpha in np.arange(0, 1 + step/2, step):
        alpha = round(alpha, 5)
        merged_df['temp_score'] = alpha * merged_df['local_accuracy'] + (1 - alpha) * merged_df['global_accuracy_mean']

        top_snps = merged_df.nlargest(min(top_n, len(merged_df)), 'temp_score')['SNP'].tolist()

        if not top_snps:
            logger.log(f"  alpha={alpha:.2f}: No SNPs selected. Skipping evaluation.")
            score = 0.0
        else:
            from .plotting import evaluate_snp_subset
            score = evaluate_snp_subset(raw_df, top_snps, y, n_splits=n_splits, logger=logger)
            logger.log(f"  alpha={alpha:.2f}, CV Accuracy={score:.4f} using {len(top_snps)} SNPs")

        alpha_list.append(alpha)
        score_list.append(score)

        if score > best_score:
            best_score = score
            best_alpha = alpha

    logger.log(f"Alpha search finished. Best alpha={best_alpha:.2f} with CV Accuracy={best_score:.4f}")
    df_alpha = pd.DataFrame({'alpha': alpha_list, 'CV_accuracy': score_list})
    return df_alpha, best_alpha, best_score
