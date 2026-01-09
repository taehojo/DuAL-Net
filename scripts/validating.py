#!/usr/bin/env python3
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")


def run_rf_tabnet_cv(X, y, n_splits=5, tabnet_epochs=10, tabnet_patience=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(y))
    oof_tab = np.zeros(len(y))

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_trn, X_val = X[tr_idx], X[va_idx]
        y_trn, y_val = y[tr_idx], y[va_idx]

        var = X_trn.var(axis=0)
        keep_idx = var > 0
        X_trn_fold = X_trn[:, keep_idx]
        X_val_fold = X_val[:, keep_idx]

        if X_trn_fold.shape[1] == 0:
            oof_rf[va_idx] = 0.5
            oof_tab[va_idx] = 0.5
            continue

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_trn_fold, y_trn)
        oof_rf[va_idx] = rf.predict_proba(X_val_fold)[:, 1]

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
        except:
            oof_tab[va_idx] = 0.5

    X_meta = np.column_stack([oof_rf, oof_tab])
    meta_lr = LogisticRegression(random_state=42)
    meta_lr.fit(X_meta, y)
    oof_ensemble = meta_lr.predict_proba(X_meta)[:, 1]

    return {
        'rf_auc': roc_auc_score(y, oof_rf),
        'tabnet_auc': roc_auc_score(y, oof_tab),
        'ensemble_auc': roc_auc_score(y, oof_ensemble),
        'ensemble_acc': accuracy_score(y, (oof_ensemble >= 0.5).astype(int))
    }


def load_rankings(rankings_file):
    df = pd.read_csv(rankings_file)
    if 'mean_score' in df.columns:
        return df.sort_values('mean_score', ascending=False)['SNP'].tolist()
    elif 'final_score' in df.columns:
        return df.sort_values('final_score', ascending=False)['SNP'].tolist()
    else:
        return df['SNP'].tolist()


def map_snp_names(ranking_snps, data_columns, rsid_to_pos=None):
    data_set = set(data_columns)
    mapped = []

    for snp in ranking_snps:
        if snp in data_set:
            mapped.append(snp)
        elif snp.startswith('rs') and rsid_to_pos and snp in rsid_to_pos:
            pos_name = f"chr19:{rsid_to_pos[snp]}"
            if pos_name in data_set:
                mapped.append(pos_name)
        elif snp.startswith('19:'):
            chr_name = 'chr' + snp
            if chr_name in data_set:
                mapped.append(chr_name)

    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings', required=True, help='Path to SNP rankings CSV (from modeling.py)')
    parser.add_argument('--raw', required=True, help='Path to validation raw genotype file (.raw)')
    parser.add_argument('--anno', default=None, help='Path to annotation file (for rsID to position mapping)')
    parser.add_argument('--output', default='./output_validation', help='Output directory')
    parser.add_argument('--top_n', type=str, default='100,500,1000', help='Comma-separated list of top N SNPs to evaluate')
    parser.add_argument('--tabnet_epochs', type=int, default=10)
    parser.add_argument('--tabnet_patience', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("DuAL-Net External Validation")
    print("=" * 60)

    print(f"Loading validation data from: {args.raw}")
    raw_df = pd.read_csv(args.raw, sep=r'\s+')
    y = (raw_df['PHENOTYPE'] == 2).astype(int).values

    drop_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns], errors='ignore', inplace=True)
    raw_df.columns = [c.split('_')[0] for c in raw_df.columns]

    print(f"  Loaded {len(y)} samples, {raw_df.shape[1]} SNPs")
    print(f"  Labels: CN={sum(y==0)}, AD={sum(y==1)}")

    rsid_to_pos = None
    if args.anno:
        print(f"Loading annotation from: {args.anno}")
        anno = pd.read_csv(args.anno)
        if 'rs_id' in anno.columns and 'pos' in anno.columns:
            rsid_to_pos = dict(zip(anno['rs_id'], anno['pos']))
            print(f"  Loaded {len(rsid_to_pos)} rsID mappings")

    print(f"Loading SNP rankings from: {args.rankings}")
    all_ranked_snps = load_rankings(args.rankings)
    print(f"  Loaded {len(all_ranked_snps)} ranked SNPs")

    top_n_list = [int(x.strip()) for x in args.top_n.split(',')]
    results_all = {}

    for n in top_n_list:
        print(f"\n{'='*60}")
        print(f"Evaluating Top {n} vs Bottom {n} SNPs")
        print("=" * 60)

        top_snps = all_ranked_snps[:n]
        bottom_snps = all_ranked_snps[-n:]

        top_mapped = map_snp_names(top_snps, raw_df.columns.tolist(), rsid_to_pos)
        bottom_mapped = map_snp_names(bottom_snps, raw_df.columns.tolist(), rsid_to_pos)

        print(f"  Top {n}: {len(top_mapped)} SNPs found in validation data")
        print(f"  Bottom {n}: {len(bottom_mapped)} SNPs found in validation data")

        if len(top_mapped) < 10:
            print(f"  Skipping Top {n} - not enough SNPs")
            continue

        X_top = np.nan_to_num(raw_df[top_mapped].values, nan=0.0)
        print(f"  Running RF+TabNet on Top {n}...")
        res_top = run_rf_tabnet_cv(X_top, y, tabnet_epochs=args.tabnet_epochs, tabnet_patience=args.tabnet_patience)

        results_all[f'top_{n}'] = {
            'n_snps_found': len(top_mapped),
            **res_top
        }

        print(f"    RF AUC: {res_top['rf_auc']:.4f}")
        print(f"    TabNet AUC: {res_top['tabnet_auc']:.4f}")
        print(f"    Ensemble AUC: {res_top['ensemble_auc']:.4f}")

        if len(bottom_mapped) >= 10:
            X_bot = np.nan_to_num(raw_df[bottom_mapped].values, nan=0.0)
            print(f"  Running RF+TabNet on Bottom {n}...")
            res_bot = run_rf_tabnet_cv(X_bot, y, tabnet_epochs=args.tabnet_epochs, tabnet_patience=args.tabnet_patience)

            results_all[f'bottom_{n}'] = {
                'n_snps_found': len(bottom_mapped),
                **res_bot
            }

            print(f"    RF AUC: {res_bot['rf_auc']:.4f}")
            print(f"    Ensemble AUC: {res_bot['ensemble_auc']:.4f}")
            print(f"    Diff (Top - Bottom): {res_top['ensemble_auc'] - res_bot['ensemble_auc']:+.4f}")

    output_file = os.path.join(args.output, "validation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'N':>6} | {'Top AUC':>10} | {'Bottom AUC':>10} | {'Diff':>10}")
    print("-" * 45)

    for n in top_n_list:
        top_key = f'top_{n}'
        bot_key = f'bottom_{n}'
        if top_key in results_all:
            top_auc = results_all[top_key]['ensemble_auc']
            if bot_key in results_all:
                bot_auc = results_all[bot_key]['ensemble_auc']
                diff = top_auc - bot_auc
                print(f"{n:>6} | {top_auc:>10.4f} | {bot_auc:>10.4f} | {diff:>+10.4f}")
            else:
                print(f"{n:>6} | {top_auc:>10.4f} | {'N/A':>10} | {'N/A':>10}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
