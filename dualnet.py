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

ANNOTATION_WEIGHTS = {
    'established_risk_allele': 10.0,
    'pathogenic': 5.0,
    'likely_pathogenic': 4.0,
    'risk_factor': 3.0,
    'drug_response': 2.0,
    'protective': 1.5,
    'association': 1.0,
}

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
    n_samples, n_snps = raw_df.shape
    n_windows = n_snps // window_size
    snp_scores = {}
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
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
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_tr, y_tr)
            accs.append(rf.score(X_val, y_val))
        window_acc = np.mean(accs)
        for snp in window_snps:
            snp_scores[snp] = window_acc
    if n_snps % window_size > 0:
        for snp in raw_df.columns[n_windows * window_size:]:
            snp_scores[snp] = 0.5
    return snp_scores

def compute_global_scores(raw_df, anno_df):
    global_scores = {snp: 0.0 for snp in raw_df.columns}
    for _, row in anno_df.iterrows():
        snp = row.get('rs_id', row.get('SNP', None))
        if snp is None or snp not in global_scores:
            continue
        score = sum(weight for feature, weight in ANNOTATION_WEIGHTS.items() if feature in row and row[feature] == 1)
        global_scores[snp] = score
    max_score = max(global_scores.values()) if max(global_scores.values()) > 0 else 1
    return {snp: score / max_score for snp, score in global_scores.items()}

def compute_combined_scores(local_scores, global_scores, alpha):
    return {snp: alpha * local_scores[snp] + (1 - alpha) * global_scores.get(snp, 0.0) for snp in local_scores}

def rf_tabnet_ensemble(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(X_train))
    oof_tab = np.zeros(len(X_train))
    test_rf = np.zeros(len(X_test))
    test_tab = np.zeros(len(X_test))
    for tr_idx, val_idx in kf.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
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

def run_dualnet(raw_file, dx_file, annotation_file, output_dir, alpha=0.6, window_size=100):
    os.makedirs(output_dir, exist_ok=True)
    raw_df, y = load_data(raw_file, dx_file)
    anno_df = load_annotations(annotation_file)
    print(f"Loaded {len(y)} samples, {raw_df.shape[1]} SNPs")
    subset_sizes = [100, 500, 1000]
    results = {size: {'top': [], 'bottom': []} for size in subset_sizes}
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), 1):
        print(f"Fold {fold}/5")
        local_scores = compute_local_scores(raw_df.iloc[train_idx], y[train_idx], window_size)
        global_scores = compute_global_scores(raw_df, anno_df)
        combined_scores = compute_combined_scores(local_scores, global_scores, alpha)
        ranked_snps = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        if 'rs429358' in ranked_snps:
            print(f"  rs429358 rank: {ranked_snps.index('rs429358') + 1}")
        for size in subset_sizes:
            auc_top = evaluate_snp_subset(raw_df, ranked_snps[:size], y, train_idx, test_idx)
            auc_bottom = evaluate_snp_subset(raw_df, ranked_snps[-size:], y, train_idx, test_idx)
            results[size]['top'].append(auc_top)
            results[size]['bottom'].append(auc_bottom)
            print(f"  {size} SNPs - Top: {auc_top:.3f}, Bottom: {auc_bottom:.3f}")
    print("\nSummary:")
    summary = {}
    for size in subset_sizes:
        summary[size] = {
            'top': {'mean': np.mean(results[size]['top']), 'std': np.std(results[size]['top'])},
            'bottom': {'mean': np.mean(results[size]['bottom']), 'std': np.std(results[size]['bottom'])}
        }
        print(f"{size} SNPs - Top: {summary[size]['top']['mean']:.3f}±{summary[size]['top']['std']:.3f}, "
              f"Bottom: {summary[size]['bottom']['mean']:.3f}±{summary[size]['bottom']['std']:.3f}")
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({'results': {str(k): v for k, v in results.items()}, 'summary': {str(k): v for k, v in summary.items()}}, f, indent=2)
    ranked_df = pd.DataFrame({'SNP': ranked_snps, 'score': [combined_scores[s] for s in ranked_snps]})
    ranked_df.to_csv(os.path.join(output_dir, 'ranked_snps.csv'), index=False)
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Raw SNP data file (.raw)")
    parser.add_argument("--dx", required=True, help="Diagnosis file")
    parser.add_argument("--annotation", required=True, help="Annotation CSV file")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.6, help="Integration weight (default: 0.6)")
    parser.add_argument("--window", type=int, default=100, help="Window size (default: 100)")
    args = parser.parse_args()
    run_dualnet(args.raw, args.dx, args.annotation, args.output, args.alpha, args.window)
