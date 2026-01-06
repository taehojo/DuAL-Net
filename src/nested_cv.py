#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested Cross-Validation for DuAL-Net

Implements proper nested CV to avoid optimistic bias:
- Outer loop (5-fold): Final model evaluation on held-out test set
- Inner loop (5-fold): Alpha optimization using only training data

This prevents data leakage from alpha selection to final evaluation.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .utils import Logger, split_into_windows_as_sequence, calculate_95_ci, format_ci_string
from .modeling import oof_stacking_rf_tabnet


def inner_alpha_search(merged_df, raw_df, y_train, train_indices,
                       top_n=100, n_inner_splits=5, alpha_step=0.1, logger=None):
    """
    Inner loop alpha search using only training data.

    Args:
        merged_df: DataFrame with SNP, local_accuracy, global_accuracy_mean
        raw_df: Raw genotype data (full dataset, will be indexed by train_indices)
        y_train: Training labels (already subset)
        train_indices: Indices into raw_df for training samples
        top_n: Number of top SNPs to select
        n_inner_splits: Number of inner CV folds
        alpha_step: Step size for alpha search
        logger: Logger object

    Returns:
        best_alpha, best_cv_score, alpha_results (DataFrame)
    """
    if logger:
        logger.log(f"    Inner alpha search: {n_inner_splits}-fold CV, top_n={top_n}")

    alpha_list = []
    score_list = []
    best_alpha = 0.0
    best_score = -1.0

    # Get training subset of raw_df
    raw_df_train = raw_df.iloc[train_indices].reset_index(drop=True)

    for alpha in np.arange(0, 1 + alpha_step/2, alpha_step):
        alpha = round(alpha, 3)

        # Calculate combined score
        merged_df['temp_score'] = (
            alpha * merged_df['local_accuracy'] +
            (1 - alpha) * merged_df['global_accuracy_mean']
        )

        # Select top SNPs
        top_snps = merged_df.nlargest(min(top_n, len(merged_df)), 'temp_score')['SNP'].tolist()
        valid_snps = [snp for snp in top_snps if snp in raw_df_train.columns]

        if not valid_snps:
            alpha_list.append(alpha)
            score_list.append(0.0)
            continue

        # Inner CV on training data only
        X_inner = raw_df_train[valid_snps].values
        kf_inner = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)

        inner_accs = []
        for tr_idx, val_idx in kf_inner.split(X_inner, y_train):
            X_tr, X_val = X_inner[tr_idx], X_inner[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr_scaled, y_tr)
            preds = rf.predict(X_val_scaled)
            inner_accs.append(accuracy_score(y_val, preds))

        cv_score = np.mean(inner_accs) if inner_accs else 0.0
        alpha_list.append(alpha)
        score_list.append(cv_score)

        if cv_score > best_score:
            best_score = cv_score
            best_alpha = alpha

    if logger:
        logger.log(f"    Inner best alpha={best_alpha:.2f}, CV accuracy={best_score:.4f}")

    df_alpha = pd.DataFrame({'alpha': alpha_list, 'inner_cv_accuracy': score_list})
    return best_alpha, best_score, df_alpha


def compute_local_scores_for_fold(raw_df, y, train_indices, window_size,
                                  tabnet_epochs, tabnet_patience, n_splits, logger):
    """
    Compute local window scores using only training data from outer fold.
    Uses RF+TabNet stacking ensemble (original DuAL-Net methodology).

    Returns:
        DataFrame with SNP, window_id, local_accuracy
    """
    from sklearn.model_selection import train_test_split

    raw_df_train = raw_df.iloc[train_indices].reset_index(drop=True)
    y_train = y[train_indices]

    X_seq = split_into_windows_as_sequence(raw_df_train, window_size)
    if X_seq.size == 0:
        logger.log("    No windows created. Returning empty local results.")
        return pd.DataFrame({'SNP': [], 'window_id': [], 'local_accuracy': []})

    N, W, dim = X_seq.shape
    logger.log(f"    Local analysis: N={N}, W={W} windows (using RF+TabNet stacking)")

    acc_dict = {}
    wdict = {}

    for w in range(W):
        X_window = X_seq[:, w]

        # Train/test split for this window (same as original analysis.py)
        X_win_train, X_win_test, y_win_train, y_win_test = train_test_split(
            X_window, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        scaler = StandardScaler()
        X_win_train_scaled = scaler.fit_transform(X_win_train)
        X_win_test_scaled = scaler.transform(X_win_test)

        if X_win_train_scaled.shape[0] < n_splits * 2 or len(np.unique(y_win_train)) < 2:
            logger.log(f"    Window {w+1}: Insufficient data. Accuracy set to 0.5.")
            acc_win = 0.5
        else:
            # Use RF+TabNet stacking (original methodology)
            test_probs = oof_stacking_rf_tabnet(
                X_win_train_scaled, y_win_train,
                X_win_test_scaled, y_win_test,
                logger=logger,
                n_splits=n_splits,
                tabnet_epochs=tabnet_epochs,
                tabnet_patience=tabnet_patience
            )
            test_preds = (test_probs >= 0.5).astype(int)
            acc_win = accuracy_score(y_win_test, test_preds)

        if (w + 1) % 20 == 0 or w == W - 1:
            logger.log(f"    Window {w+1}/{W} accuracy = {acc_win:.4f}")

        # Map to SNPs
        st = w * window_size
        en = st + window_size
        for snp in raw_df.columns[st:en]:
            acc_dict[snp] = acc_win
            wdict[snp] = w + 1

    df_local = pd.DataFrame({
        'SNP': list(acc_dict.keys()),
        'window_id': [wdict[s] for s in acc_dict.keys()],
        'local_accuracy': [acc_dict[s] for s in acc_dict.keys()]
    })

    return df_local


def compute_global_scores_for_fold(raw_df, y, train_indices, anno_df,
                                   tabnet_epochs, tabnet_patience, n_splits, logger):
    """
    Compute global annotation scores using only training data from outer fold.
    Uses RF+TabNet stacking ensemble (original DuAL-Net methodology).

    Returns:
        DataFrame with SNP and global_accuracy columns
    """
    from pandas.api.types import is_numeric_dtype
    from sklearn.model_selection import train_test_split

    raw_df_train = raw_df.iloc[train_indices].reset_index(drop=True)
    y_train = y[train_indices]

    if 'rs_id' not in anno_df.columns:
        logger.log("    Warning: 'rs_id' not in annotation. Returning empty global results.")
        return pd.DataFrame({'SNP': [], 'global_accuracy_mean': []})

    anno_indexed = anno_df.set_index('rs_id') if anno_df.index.name != 'rs_id' else anno_df
    numeric_cols = [c for c in anno_indexed.columns if is_numeric_dtype(anno_indexed[c])]

    # Create mapping from rsID (without suffix) to actual column name (with suffix)
    # Raw columns are like "rs429358_C", annotation has "rs429358"
    rsid_to_col = {}
    for col_name in raw_df_train.columns:
        if col_name.startswith('rs') or ':' in col_name:
            # Strip allele suffix (e.g., rs429358_C -> rs429358)
            rsid = col_name.rsplit('_', 1)[0] if '_' in col_name else col_name
            rsid_to_col[rsid] = col_name

    results_list = []
    logger.log(f"    Global analysis: {len(numeric_cols)} annotation columns (using RF+TabNet stacking)")

    for col_idx, col in enumerate(numeric_cols):
        subset_snps = anno_indexed.index[anno_indexed[col] > 0].tolist()
        # Map annotation rsIDs to actual raw data columns
        valid_snps = [rsid_to_col[snp] for snp in subset_snps if snp in rsid_to_col]

        if not valid_snps:
            continue

        X_sub = raw_df_train[valid_snps].values

        # Use train/test split with RF+TabNet stacking (same as original analysis.py)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sub, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        if X_tr_scaled.shape[0] < n_splits * 2 or len(np.unique(y_tr)) < 2:
            acc_global = 0.5
        else:
            # Use RF+TabNet stacking (original methodology)
            test_probs = oof_stacking_rf_tabnet(
                X_tr_scaled, y_tr,
                X_val_scaled, y_val,
                logger=logger,
                n_splits=n_splits,
                tabnet_epochs=tabnet_epochs,
                tabnet_patience=tabnet_patience
            )
            test_preds = (test_probs >= 0.5).astype(int)
            acc_global = accuracy_score(y_val, test_preds)

        for snp in valid_snps:
            results_list.append({'SNP': snp, f'{col}_global_accuracy': acc_global})

        if (col_idx + 1) % 10 == 0 or col_idx == len(numeric_cols) - 1:
            logger.log(f"    Annotation {col_idx+1}/{len(numeric_cols)} accuracy = {acc_global:.4f}")

    if not results_list:
        return pd.DataFrame({'SNP': [], 'global_accuracy_mean': []})

    df_global = pd.DataFrame(results_list).groupby('SNP').first().reset_index()

    # Compute mean of global accuracies
    global_cols = [c for c in df_global.columns if c.endswith('_global_accuracy')]
    if global_cols:
        df_global['global_accuracy_mean'] = df_global[global_cols].mean(axis=1)
    else:
        df_global['global_accuracy_mean'] = 0.0

    return df_global


def nested_cv_evaluation(raw_df, y, anno_df, logger,
                         n_outer_splits=5, n_inner_splits=5,
                         window_size=100, top_n=100, alpha_step=0.1,
                         tabnet_epochs=50, tabnet_patience=10,
                         result_dir="result/nested-cv"):
    """
    Perform nested cross-validation for unbiased evaluation.

    Outer loop: 5-fold CV for final test evaluation
    Inner loop: 5-fold CV for alpha optimization (on outer training data only)

    Args:
        raw_df: Raw genotype DataFrame
        y: Labels array
        anno_df: Annotation DataFrame with rs_id column
        logger: Logger object
        n_outer_splits: Number of outer CV folds
        n_inner_splits: Number of inner CV folds for alpha search
        window_size: Window size for local analysis
        top_n: Number of top SNPs to select
        alpha_step: Step size for alpha search
        tabnet_epochs: TabNet training epochs
        tabnet_patience: TabNet early stopping patience
        result_dir: Directory to save results

    Returns:
        dict with nested CV results
    """
    os.makedirs(result_dir, exist_ok=True)

    logger.log("=" * 60)
    logger.log("NESTED CROSS-VALIDATION")
    logger.log("=" * 60)
    logger.log(f"Outer folds: {n_outer_splits}")
    logger.log(f"Inner folds: {n_inner_splits}")
    logger.log(f"Window size: {window_size}")
    logger.log(f"Top N SNPs: {top_n}")
    logger.log(f"Alpha step: {alpha_step}")
    logger.log("=" * 60)

    outer_kf = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)

    outer_results = []
    all_test_aucs = []
    all_test_accs = []
    all_best_alphas = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(raw_df, y)):
        logger.log(f"\n{'='*50}")
        logger.log(f"OUTER FOLD {outer_fold + 1}/{n_outer_splits}")
        logger.log(f"{'='*50}")
        logger.log(f"Train samples: {len(outer_train_idx)}, Test samples: {len(outer_test_idx)}")

        y_outer_train = y[outer_train_idx]
        y_outer_test = y[outer_test_idx]

        # Step 1: Compute local scores on outer training data
        logger.log("\n  [Step 1] Computing local scores on outer training data...")
        df_local = compute_local_scores_for_fold(
            raw_df, y, outer_train_idx, window_size,
            tabnet_epochs, tabnet_patience, n_inner_splits, logger
        )

        # Step 2: Compute global scores on outer training data
        logger.log("\n  [Step 2] Computing global scores on outer training data...")
        df_global = compute_global_scores_for_fold(
            raw_df, y, outer_train_idx, anno_df,
            tabnet_epochs, tabnet_patience, n_inner_splits, logger
        )

        # Step 3: Merge local and global scores
        if df_global.empty or 'global_accuracy_mean' not in df_global.columns:
            df_global = pd.DataFrame({
                'SNP': df_local['SNP'].tolist(),
                'global_accuracy_mean': [0.0] * len(df_local)
            })

        merged_df = pd.merge(
            df_local[['SNP', 'local_accuracy']],
            df_global[['SNP', 'global_accuracy_mean']],
            on='SNP', how='outer'
        ).fillna(0.0)

        logger.log(f"  Merged {len(merged_df)} SNPs with local and global scores")

        # Step 4: Inner alpha search using only outer training data
        logger.log("\n  [Step 3] Inner alpha search (using only outer training data)...")
        best_alpha, inner_best_score, df_alpha_inner = inner_alpha_search(
            merged_df, raw_df, y_outer_train, outer_train_idx,
            top_n=top_n, n_inner_splits=n_inner_splits,
            alpha_step=alpha_step, logger=logger
        )

        all_best_alphas.append(best_alpha)

        # Step 5: Select top SNPs with best alpha
        merged_df['final_score'] = (
            best_alpha * merged_df['local_accuracy'] +
            (1 - best_alpha) * merged_df['global_accuracy_mean']
        )
        top_snps = merged_df.nlargest(min(top_n, len(merged_df)), 'final_score')['SNP'].tolist()
        valid_top_snps = [snp for snp in top_snps if snp in raw_df.columns]

        logger.log(f"\n  [Step 4] Selected {len(valid_top_snps)} top SNPs with alpha={best_alpha:.2f}")

        # Step 6: Train final model on outer training data, evaluate on outer test
        # Use RF+TabNet stacking (original DuAL-Net methodology)
        logger.log("\n  [Step 5] Training RF+TabNet stacking and evaluating on outer test set...")

        raw_df_outer_train = raw_df.iloc[outer_train_idx][valid_top_snps].values
        raw_df_outer_test = raw_df.iloc[outer_test_idx][valid_top_snps].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(raw_df_outer_train)
        X_test_scaled = scaler.transform(raw_df_outer_test)

        # Use RF+TabNet stacking (original DuAL-Net methodology)
        test_probs = oof_stacking_rf_tabnet(
            X_train_scaled, y_outer_train,
            X_test_scaled, y_outer_test,
            logger=logger,
            n_splits=n_inner_splits,
            tabnet_epochs=tabnet_epochs,
            tabnet_patience=tabnet_patience
        )
        test_preds = (test_probs >= 0.5).astype(int)

        test_acc = accuracy_score(y_outer_test, test_preds)
        test_auc = roc_auc_score(y_outer_test, test_probs)

        all_test_accs.append(test_acc)
        all_test_aucs.append(test_auc)

        logger.log(f"  Outer Fold {outer_fold + 1} Results:")
        logger.log(f"    Best Alpha: {best_alpha:.2f}")
        logger.log(f"    Inner CV Accuracy: {inner_best_score:.4f}")
        logger.log(f"    Outer Test Accuracy: {test_acc:.4f}")
        logger.log(f"    Outer Test AUC: {test_auc:.4f}")

        # Save fold results
        fold_result = {
            'outer_fold': outer_fold + 1,
            'n_train': len(outer_train_idx),
            'n_test': len(outer_test_idx),
            'best_alpha': best_alpha,
            'inner_cv_accuracy': inner_best_score,
            'outer_test_accuracy': test_acc,
            'outer_test_auc': test_auc,
            'n_top_snps': len(valid_top_snps)
        }
        outer_results.append(fold_result)

        # Save fold-specific files
        fold_dir = os.path.join(result_dir, f"fold_{outer_fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        merged_df.to_csv(os.path.join(fold_dir, "merged_scores.csv"), index=False)
        df_alpha_inner.to_csv(os.path.join(fold_dir, "alpha_search.csv"), index=False)
        pd.DataFrame({'SNP': valid_top_snps}).to_csv(
            os.path.join(fold_dir, "top_snps.csv"), index=False
        )

    # Compute summary statistics
    logger.log("\n" + "=" * 60)
    logger.log("NESTED CV SUMMARY")
    logger.log("=" * 60)

    ci_auc = calculate_95_ci(all_test_aucs)
    ci_acc = calculate_95_ci(all_test_accs)

    logger.log(f"\nOuter Fold Results:")
    for i, result in enumerate(outer_results):
        logger.log(f"  Fold {i+1}: Alpha={result['best_alpha']:.2f}, "
                   f"Test ACC={result['outer_test_accuracy']:.4f}, "
                   f"Test AUC={result['outer_test_auc']:.4f}")

    logger.log(f"\nMean Best Alpha: {np.mean(all_best_alphas):.2f} +/- {np.std(all_best_alphas):.2f}")
    logger.log(f"\nTest AUC: {format_ci_string(ci_auc)}")
    logger.log(f"Test Accuracy: {format_ci_string(ci_acc)}")
    logger.log("=" * 60)

    # Save overall summary
    summary = {
        'n_outer_splits': n_outer_splits,
        'n_inner_splits': n_inner_splits,
        'window_size': window_size,
        'top_n': top_n,
        'alpha_step': alpha_step,
        'fold_results': outer_results,
        'summary': {
            'test_auc_mean': ci_auc['mean'],
            'test_auc_std': ci_auc['std'],
            'test_auc_ci_lower': ci_auc['ci_lower'],
            'test_auc_ci_upper': ci_auc['ci_upper'],
            'test_acc_mean': ci_acc['mean'],
            'test_acc_std': ci_acc['std'],
            'test_acc_ci_lower': ci_acc['ci_lower'],
            'test_acc_ci_upper': ci_acc['ci_upper'],
            'mean_best_alpha': float(np.mean(all_best_alphas)),
            'std_best_alpha': float(np.std(all_best_alphas)),
            'all_test_aucs': all_test_aucs,
            'all_test_accs': all_test_accs,
            'all_best_alphas': all_best_alphas
        }
    }

    with open(os.path.join(result_dir, "nested_cv_results.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    logger.log(f"\nResults saved to {result_dir}/")

    return summary
