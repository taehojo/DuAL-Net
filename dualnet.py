#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

def load_data(raw_file, dx_file):
    raw_df = pd.read_csv(raw_file, sep=r'\s+')
    drop_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns], errors='ignore', inplace=True)
    raw_df.columns = [c.split('_')[0] for c in raw_df.columns]
    dx_df = pd.read_csv(dx_file, sep=r'\s+')
    y = dx_df['New_Label'].values if 'New_Label' in dx_df.columns else dx_df.iloc[:, -1].values
    return raw_df, y

def load_annotations(annotation_file):
    return pd.read_csv(annotation_file)

def compute_local_scores(raw_df, y, window_size=100):
    n_snps = raw_df.shape[1]
    n_windows = n_snps // window_size
    snp_scores = {}
    for w in range(n_windows):
        start, end = w * window_size, (w + 1) * window_size
        window_snps = raw_df.columns[start:end].tolist()
        X_window = raw_df.iloc[:, start:end].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tr_idx, val_idx in kf.split(X_window):
            X_tr, X_val = X_window[tr_idx], X_window[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            accs.append(rf.score(X_val, y_val))
        window_acc = np.mean(accs)
        for snp in window_snps:
            snp_scores[snp] = window_acc
    for snp in raw_df.columns[n_windows * window_size:]:
        snp_scores[snp] = 0.5
    return snp_scores

def compute_global_scores(raw_df, y, anno_df):
    anno_df = anno_df.copy()
    if 'rs_id' in anno_df.columns:
        anno_df = anno_df.set_index('rs_id')
    numeric_cols = [c for c in anno_df.columns if anno_df[c].dtype in ['int64', 'float64']]
    snp_global_accs = {snp: [] for snp in raw_df.columns}
    for col in numeric_cols:
        subset_snps = anno_df.index[anno_df[col] > 0].tolist()
        valid_snps = [s for s in subset_snps if s in raw_df.columns]
        if len(valid_snps) < 5:
            continue
        X_sub = raw_df[valid_snps].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tr_idx, val_idx in kf.split(X_sub):
            X_tr, X_val = X_sub[tr_idx], X_sub[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            accs.append(rf.score(X_val, y_val))
        group_acc = np.mean(accs)
        for snp in valid_snps:
            snp_global_accs[snp].append(group_acc)
    return {snp: np.mean(accs) if accs else 0.0 for snp, accs in snp_global_accs.items()}

def compute_combined_scores(local_scores, global_scores, alpha):
    return {snp: alpha * local_scores[snp] + (1 - alpha) * global_scores.get(snp, 0.0) for snp in local_scores}

def rf_tabnet_ensemble(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_rf, oof_tab = np.zeros(len(X_train)), np.zeros(len(X_train))
    test_rf, test_tab = np.zeros(len(X_test)), np.zeros(len(X_test))
    for tr_idx, val_idx in kf.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
        test_rf += rf.predict_proba(X_test_scaled)[:, 1] / 5
        if X_tr.shape[1] >= 4:
            try:
                tab = TabNetClassifier(n_d=8, n_a=8, n_steps=3, optimizer_fn=torch.optim.Adam,
                                       optimizer_params=dict(lr=1e-3), mask_type="entmax", verbose=0)
                tab.fit(X_tr, y_tr, max_epochs=10, patience=5, eval_set=[(X_val, y_val)], eval_metric=['auc'])
                oof_tab[val_idx] = tab.predict_proba(X_val)[:, 1]
                test_tab += tab.predict_proba(X_test_scaled)[:, 1] / 5
            except:
                oof_tab[val_idx] = 0.5
                test_tab += 0.1
        else:
            oof_tab[val_idx] = 0.5
            test_tab += 0.1
    meta = LogisticRegression(random_state=42)
    meta.fit(np.column_stack([oof_rf, oof_tab]), y_train)
    return meta.predict_proba(np.column_stack([test_rf, test_tab]))[:, 1]

def evaluate_snp_subset(raw_df, snps, y, train_idx, test_idx):
    valid_snps = [s for s in snps if s in raw_df.columns]
    if len(valid_snps) < 4:
        return 0.5
    X = raw_df[valid_snps].values
    return roc_auc_score(y[test_idx], rf_tabnet_ensemble(X[train_idx], y[train_idx], X[test_idx]))

def find_best_alpha(raw_df, local_scores, global_scores, y, train_idx):
    best_alpha, best_score = 0.5, 0.0
    y_train = y[train_idx]
    raw_train = raw_df.iloc[train_idx]
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for alpha in np.arange(0.0, 1.1, 0.1):
        alpha = round(alpha, 1)
        combined = compute_combined_scores(local_scores, global_scores, alpha)
        ranked = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:100]
        valid_snps = [s for s in ranked if s in raw_train.columns]
        if len(valid_snps) < 10:
            continue
        X_sub = raw_train[valid_snps].values
        accs = []
        for tr_idx, val_idx in inner_cv.split(X_sub, y_train):
            X_tr, X_val = X_sub[tr_idx], X_sub[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            accs.append(rf.score(X_val, y_val))
        score = np.mean(accs)
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha

def run_dualnet(raw_file, dx_file, annotation_file, output_dir, window_size=100):
    os.makedirs(output_dir, exist_ok=True)
    raw_df, y = load_data(raw_file, dx_file)
    anno_df = load_annotations(annotation_file)
    print(f"Loaded {len(y)} samples, {raw_df.shape[1]} SNPs")
    subset_sizes = [100, 500, 1000]
    results = {size: {'top': [], 'bottom': []} for size in subset_sizes}
    alpha_values = []
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), 1):
        print(f"\nFold {fold}/5")
        raw_train, y_train = raw_df.iloc[train_idx], y[train_idx]
        local_scores = compute_local_scores(raw_train, y_train, window_size)
        global_scores = compute_global_scores(raw_train, y_train, anno_df)
        best_alpha = find_best_alpha(raw_df, local_scores, global_scores, y, train_idx)
        alpha_values.append(best_alpha)
        print(f"  Best alpha: {best_alpha}")
        combined_scores = compute_combined_scores(local_scores, global_scores, best_alpha)
        ranked_snps = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        if 'rs429358' in ranked_snps:
            print(f"  rs429358 rank: {ranked_snps.index('rs429358') + 1}")
        for size in subset_sizes:
            auc_top = evaluate_snp_subset(raw_df, ranked_snps[:size], y, train_idx, test_idx)
            auc_bottom = evaluate_snp_subset(raw_df, ranked_snps[-size:], y, train_idx, test_idx)
            results[size]['top'].append(auc_top)
            results[size]['bottom'].append(auc_bottom)
            print(f"  {size} SNPs - Top: {auc_top:.3f}, Bottom: {auc_bottom:.3f}")
    print(f"\nMean alpha: {np.mean(alpha_values):.2f} +/- {np.std(alpha_values):.2f}")
    print("\nSummary:")
    summary = {}
    for size in subset_sizes:
        summary[size] = {
            'top': {'mean': np.mean(results[size]['top']), 'std': np.std(results[size]['top'])},
            'bottom': {'mean': np.mean(results[size]['bottom']), 'std': np.std(results[size]['bottom'])}
        }
        gap = summary[size]['top']['mean'] - summary[size]['bottom']['mean']
        print(f"{size} SNPs - Top: {summary[size]['top']['mean']:.3f}+/-{summary[size]['top']['std']:.3f}, "
              f"Bottom: {summary[size]['bottom']['mean']:.3f}+/-{summary[size]['bottom']['std']:.3f}, Gap: {gap:.3f}")
    output = {
        'alpha_values': alpha_values,
        'alpha_mean': float(np.mean(alpha_values)),
        'results': {str(k): v for k, v in results.items()},
        'summary': {str(k): v for k, v in summary.items()}
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuAL-Net: Dual-network SNP prioritization")
    parser.add_argument("--raw", required=True, help="Raw SNP data file (.raw)")
    parser.add_argument("--dx", required=True, help="Diagnosis file")
    parser.add_argument("--annotation", required=True, help="Annotation CSV file")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--window", type=int, default=100, help="Window size (default: 100)")
    args = parser.parse_args()
    run_dualnet(args.raw, args.dx, args.annotation, args.output, args.window)
