#!/usr/bin/env python3
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore")


class Logger:
    def __init__(self, log_file=None):
        self.logs = []
        self.log_file = log_file

    def log(self, msg):
        print(msg)
        self.logs.append(str(msg))

    def write(self):
        if self.log_file:
            with open(self.log_file, 'w') as f:
                for line in self.logs:
                    f.write(line + "\n")


def get_zero_var_idx(X):
    if isinstance(X, np.ndarray):
        var_ = X.var(axis=0)
        return np.where(var_ == 0)[0]
    return []


def remove_zv_by_idx(X, zero_idx):
    if len(zero_idx) > 0 and isinstance(X, np.ndarray):
        return np.delete(X, zero_idx, axis=1)
    return X


def split_into_windows_as_sequence(data, window_size):
    arr = data.values
    N, M = arr.shape
    W = M // window_size
    if W == 0:
        return np.empty((N, 0, window_size))
    arr = arr[:, :W * window_size]
    seqs = [arr[:, i*window_size:(i+1)*window_size] for i in range(W)]
    return np.stack(seqs, axis=1)


def calculate_95_ci(values):
    values = [v for v in values if not np.isnan(v)]
    n = len(values)
    if n == 0:
        return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    mean = np.mean(values)
    if n < 2:
        return {'mean': mean, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    return {'mean': mean, 'std': std, 'ci_lower': mean - t_crit * se, 'ci_upper': mean + t_crit * se}


def oof_stacking_rf_tabnet(X_train, y_train, X_test, y_test, n_splits=5, tabnet_epochs=10, tabnet_patience=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(X_train))
    oof_tab = np.zeros(len(X_train))
    test_preds_rf = np.zeros((len(X_test), n_splits))
    test_preds_tab = np.zeros((len(X_test), n_splits))

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_trn, X_val = X_train[tr_idx], X_train[va_idx]
        y_trn, y_val = y_train[tr_idx], y_train[va_idx]

        zv_idx = get_zero_var_idx(X_trn)
        X_trn_fold = remove_zv_by_idx(X_trn, zv_idx)
        X_val_fold = remove_zv_by_idx(X_val, zv_idx)
        X_test_fold = remove_zv_by_idx(X_test, zv_idx)

        if len(np.unique(y_trn)) < 2:
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
        except:
            oof_rf[va_idx] = 0.5
            test_preds_rf[:, fold_i] = 0.5

        try:
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
            oof_tab[va_idx] = tab.predict_proba(X_val_fold)[:, 1]
            test_preds_tab[:, fold_i] = tab.predict_proba(X_test_fold)[:, 1]
        except:
            oof_tab[va_idx] = 0.5
            test_preds_tab[:, fold_i] = 0.5

    test_pred_rf_mean = np.mean(test_preds_rf, axis=1)
    test_pred_tab_mean = np.mean(test_preds_tab, axis=1)

    meta_model = LogisticRegression(random_state=42)
    X_meta_train = np.column_stack([oof_rf, oof_tab])
    meta_model.fit(X_meta_train, y_train)

    X_meta_test = np.column_stack([test_pred_rf_mean, test_pred_tab_mean])
    return meta_model.predict_proba(X_meta_test)[:, 1]


def compute_local_scores(raw_df, y, train_indices, window_size, n_splits, tabnet_epochs, tabnet_patience, logger):
    raw_df_train = raw_df.iloc[train_indices].reset_index(drop=True)
    y_train = y[train_indices]

    X_seq = split_into_windows_as_sequence(raw_df_train, window_size)
    if X_seq.size == 0:
        return pd.DataFrame({'SNP': [], 'window_id': [], 'local_accuracy': []})

    N, W, dim = X_seq.shape
    logger.log(f"    Local analysis: {W} windows")

    acc_dict, wdict = {}, {}
    for w in range(W):
        X_window = X_seq[:, w]
        X_win_train, X_win_test, y_win_train, y_win_test = train_test_split(
            X_window, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        scaler = StandardScaler()
        X_win_train_scaled = scaler.fit_transform(X_win_train)
        X_win_test_scaled = scaler.transform(X_win_test)

        if X_win_train_scaled.shape[0] < n_splits * 2 or len(np.unique(y_win_train)) < 2:
            acc_win = 0.5
        else:
            test_probs = oof_stacking_rf_tabnet(X_win_train_scaled, y_win_train, X_win_test_scaled, y_win_test, n_splits, tabnet_epochs, tabnet_patience)
            acc_win = accuracy_score(y_win_test, (test_probs >= 0.5).astype(int))

        if (w + 1) % 20 == 0 or w == W - 1:
            logger.log(f"    Window {w+1}/{W} accuracy = {acc_win:.4f}")

        for snp in raw_df.columns[w*window_size:(w+1)*window_size]:
            acc_dict[snp] = acc_win
            wdict[snp] = w + 1

    return pd.DataFrame({'SNP': list(acc_dict.keys()), 'window_id': [wdict[s] for s in acc_dict.keys()], 'local_accuracy': [acc_dict[s] for s in acc_dict.keys()]})


def compute_global_scores(raw_df, y, train_indices, anno_df, n_splits, tabnet_epochs, tabnet_patience, logger):
    raw_df_train = raw_df.iloc[train_indices].reset_index(drop=True)
    y_train = y[train_indices]

    if 'rs_id' not in anno_df.columns:
        return pd.DataFrame({'SNP': [], 'global_accuracy_mean': []})

    anno_indexed = anno_df.set_index('rs_id')
    numeric_cols = [c for c in anno_indexed.columns if is_numeric_dtype(anno_indexed[c])]

    rsid_to_col = {}
    for col_name in raw_df_train.columns:
        rsid = col_name.rsplit('_', 1)[0] if '_' in col_name else col_name
        rsid_to_col[rsid] = col_name

    results_list = []
    logger.log(f"    Global analysis: {len(numeric_cols)} annotation columns")

    for col_idx, col in enumerate(numeric_cols):
        subset_snps = anno_indexed.index[anno_indexed[col] > 0].tolist()
        valid_snps = [rsid_to_col[snp] for snp in subset_snps if snp in rsid_to_col]
        if not valid_snps:
            continue

        X_sub = raw_df_train[valid_snps].values
        X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y_train, test_size=0.2, random_state=42, stratify=y_train)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        if X_tr_scaled.shape[0] < n_splits * 2 or len(np.unique(y_tr)) < 2:
            acc_global = 0.5
        else:
            test_probs = oof_stacking_rf_tabnet(X_tr_scaled, y_tr, X_val_scaled, y_val, n_splits, tabnet_epochs, tabnet_patience)
            acc_global = accuracy_score(y_val, (test_probs >= 0.5).astype(int))

        for snp in valid_snps:
            results_list.append({'SNP': snp, f'{col}_global_accuracy': acc_global})

    if not results_list:
        return pd.DataFrame({'SNP': [], 'global_accuracy_mean': []})

    df_global = pd.DataFrame(results_list).groupby('SNP').first().reset_index()
    global_cols = [c for c in df_global.columns if c.endswith('_global_accuracy')]
    df_global['global_accuracy_mean'] = df_global[global_cols].mean(axis=1) if global_cols else 0.0
    return df_global


def inner_alpha_search(merged_df, raw_df, y_train, train_indices, top_n, n_inner_splits, alpha_step, tabnet_epochs, tabnet_patience, logger):
    raw_df_train = raw_df.iloc[train_indices].reset_index(drop=True)
    best_alpha, best_score = 0.0, -1.0
    alpha_results = []

    for alpha in np.arange(0, 1 + alpha_step/2, alpha_step):
        alpha = round(alpha, 3)
        merged_df['temp_score'] = alpha * merged_df['local_accuracy'] + (1 - alpha) * merged_df['global_accuracy_mean']
        top_snps = merged_df.nlargest(min(top_n, len(merged_df)), 'temp_score')['SNP'].tolist()
        valid_snps = [snp for snp in top_snps if snp in raw_df_train.columns]

        if not valid_snps:
            alpha_results.append({'alpha': alpha, 'cv_accuracy': 0.0})
            continue

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
            val_probs = oof_stacking_rf_tabnet(X_tr_scaled, y_tr, X_val_scaled, y_val, n_inner_splits, tabnet_epochs, tabnet_patience)
            inner_accs.append(accuracy_score(y_val, (val_probs >= 0.5).astype(int)))

        cv_score = np.mean(inner_accs) if inner_accs else 0.0
        alpha_results.append({'alpha': alpha, 'cv_accuracy': cv_score})
        if cv_score > best_score:
            best_score = cv_score
            best_alpha = alpha

    logger.log(f"    Best alpha={best_alpha:.2f}, CV accuracy={best_score:.4f}")
    return best_alpha, pd.DataFrame(alpha_results)


def run_nested_cv(raw_df, y, anno_df, output_dir, n_outer=5, n_inner=5, window_size=100, top_n=100, alpha_step=0.1, tabnet_epochs=10, tabnet_patience=5):
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(os.path.join(output_dir, 'modeling_log.txt'))

    logger.log("=" * 60)
    logger.log("DuAL-Net Nested Cross-Validation")
    logger.log("=" * 60)

    outer_kf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    all_fold_rankings = []
    all_test_aucs, all_test_accs, all_best_alphas = [], [], []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(raw_df, y)):
        logger.log(f"\nOUTER FOLD {outer_fold + 1}/{n_outer}")
        y_train, y_test = y[train_idx], y[test_idx]

        df_local = compute_local_scores(raw_df, y, train_idx, window_size, n_inner, tabnet_epochs, tabnet_patience, logger)
        df_global = compute_global_scores(raw_df, y, train_idx, anno_df, n_inner, tabnet_epochs, tabnet_patience, logger)

        if df_global.empty:
            df_global = pd.DataFrame({'SNP': df_local['SNP'].tolist(), 'global_accuracy_mean': [0.0] * len(df_local)})

        merged_df = pd.merge(df_local[['SNP', 'local_accuracy']], df_global[['SNP', 'global_accuracy_mean']], on='SNP', how='outer').fillna(0.0)

        best_alpha, df_alpha = inner_alpha_search(merged_df, raw_df, y_train, train_idx, top_n, n_inner, alpha_step, tabnet_epochs, tabnet_patience, logger)
        all_best_alphas.append(best_alpha)

        merged_df['final_score'] = best_alpha * merged_df['local_accuracy'] + (1 - best_alpha) * merged_df['global_accuracy_mean']
        top_snps = merged_df.nlargest(min(top_n, len(merged_df)), 'final_score')['SNP'].tolist()
        valid_top_snps = [snp for snp in top_snps if snp in raw_df.columns]

        X_train = raw_df.iloc[train_idx][valid_top_snps].values
        X_test = raw_df.iloc[test_idx][valid_top_snps].values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        test_probs = oof_stacking_rf_tabnet(X_train_scaled, y_train, X_test_scaled, y_test, n_inner, tabnet_epochs, tabnet_patience)
        test_auc = roc_auc_score(y_test, test_probs)
        test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
        all_test_aucs.append(test_auc)
        all_test_accs.append(test_acc)

        logger.log(f"  Fold {outer_fold+1}: AUC={test_auc:.4f}, ACC={test_acc:.4f}, Alpha={best_alpha:.2f}")

        fold_dir = os.path.join(output_dir, f"fold_{outer_fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        merged_df.to_csv(os.path.join(fold_dir, "merged_scores.csv"), index=False)
        all_fold_rankings.append(merged_df[['SNP', 'final_score']])

    consensus_df = all_fold_rankings[0].copy()
    for i, df in enumerate(all_fold_rankings[1:], 2):
        consensus_df = consensus_df.merge(df, on='SNP', how='outer', suffixes=('', f'_{i}'))
    score_cols = [c for c in consensus_df.columns if 'final_score' in c]
    consensus_df['mean_score'] = consensus_df[score_cols].mean(axis=1)
    consensus_df = consensus_df.sort_values('mean_score', ascending=False)
    consensus_df[['SNP', 'mean_score']].to_csv(os.path.join(output_dir, "consensus_snp_rankings.csv"), index=False)

    ci_auc = calculate_95_ci(all_test_aucs)
    ci_acc = calculate_95_ci(all_test_accs)

    summary = {
        'mean_alpha': float(np.mean(all_best_alphas)),
        'auc_mean': ci_auc['mean'],
        'auc_ci_lower': ci_auc['ci_lower'],
        'auc_ci_upper': ci_auc['ci_upper'],
        'acc_mean': ci_acc['mean'],
        'acc_ci_lower': ci_acc['ci_lower'],
        'acc_ci_upper': ci_acc['ci_upper']
    }

    with open(os.path.join(output_dir, "modeling_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.log("\n" + "=" * 60)
    logger.log(f"Mean Alpha: {summary['mean_alpha']:.2f}")
    logger.log(f"AUC: {ci_auc['mean']:.4f} (95% CI: {ci_auc['ci_lower']:.4f}-{ci_auc['ci_upper']:.4f})")
    logger.log(f"ACC: {ci_acc['mean']:.4f} (95% CI: {ci_acc['ci_lower']:.4f}-{ci_acc['ci_upper']:.4f})")
    logger.log("=" * 60)
    logger.write()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, help='Path to raw genotype file (.raw)')
    parser.add_argument('--dx', required=True, help='Path to phenotype file')
    parser.add_argument('--anno', required=True, help='Path to annotation file')
    parser.add_argument('--output', default='./output_modeling', help='Output directory')
    parser.add_argument('--n_outer', type=int, default=5)
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--alpha_step', type=float, default=0.1)
    parser.add_argument('--tabnet_epochs', type=int, default=10)
    parser.add_argument('--tabnet_patience', type=int, default=5)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.raw, sep=r'\s+')
    drop_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns], errors='ignore', inplace=True)
    raw_df.columns = [c.split('_')[0] for c in raw_df.columns]

    dx_df = pd.read_csv(args.dx, sep=r'\s+')
    y = dx_df['New_Label'].values

    anno_df = pd.read_csv(args.anno)

    print(f"Loaded {len(y)} samples, {raw_df.shape[1]} SNPs")
    run_nested_cv(raw_df, y, anno_df, args.output, args.n_outer, args.n_inner, args.window_size, args.top_n, args.alpha_step, args.tabnet_epochs, args.tabnet_patience)


if __name__ == "__main__":
    main()
