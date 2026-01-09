#!/usr/bin/env python3


import sys
import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

warnings.filterwarnings("ignore")


NESTED_CV_DIR = "/N/project/AiLab/R4-ADNI-1566/revise/result/nested-cv-rftabnet"
ADC_RAW_FILE = "/N/project/AiLab/R4-ADNI-1566/revise/external-validation/adc_nhw_apoe_dedup.raw"
OUTPUT_DIR = "/N/project/AiLab/R4-ADNI-1566/revise/external-validation"

TOP_N_LIST = [100, 500, 1000]
N_CV_FOLDS = 5


def load_adc_data(raw_file):
    
    print(f"Loading ADC data from: {raw_file}")
    df = pd.read_csv(raw_file, sep=r'\s+')

    
    y = (df['PHENOTYPE'] == 2).astype(int).values

    
    drop_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    
    X_df.columns = [c.split('_')[0] for c in X_df.columns]

    print(f"  Loaded {len(y)} samples, {X_df.shape[1]} SNPs")
    print(f"  Label distribution: Control={sum(y==0)}, AD={sum(y==1)}")

    return X_df, y


def aggregate_snp_rankings(nested_cv_dir, n_folds=5, rsid_to_pos=None):
    
    print("Aggregating SNP rankings across folds...")

    all_scores = []
    for fold in range(1, n_folds + 1):
        score_file = os.path.join(nested_cv_dir, f"fold_{fold}", "merged_scores.csv")
        if os.path.exists(score_file):
            df = pd.read_csv(score_file)

            
            if rsid_to_pos:
                converted = []
                for snp in df['SNP']:
                    if snp.startswith('rs') and snp in rsid_to_pos:
                        converted.append(f"19:{rsid_to_pos[snp]}")
                    else:
                        converted.append(snp)
                df['SNP'] = converted

            df = df.rename(columns={'final_score': f'score_fold{fold}'})
            all_scores.append(df[['SNP', f'score_fold{fold}']])
            print(f"  Fold {fold}: {len(df)} SNPs")

    
    merged = all_scores[0]
    for df in all_scores[1:]:
        merged = merged.merge(df, on='SNP', how='outer')

    
    score_cols = [c for c in merged.columns if c.startswith('score_fold')]
    merged['mean_score'] = merged[score_cols].mean(axis=1)
    merged['std_score'] = merged[score_cols].std(axis=1)

    
    merged = merged.sort_values('mean_score', ascending=False)

    print(f"  Total unique SNPs: {len(merged)}")

    return merged


def load_rsid_to_position_map(annotation_file="/N/project/AiLab/R4-ADNI-1566/for-Eunhye/el_snpinfo_1211.csv"):
    
    print(f"Loading rsID-to-position mapping from: {annotation_file}")
    anno = pd.read_csv(annotation_file)
    rsid_to_pos = dict(zip(anno['rs_id'], anno['pos']))
    print(f"  Loaded {len(rsid_to_pos)} rsID mappings")
    return rsid_to_pos


def map_snp_names(adni_snps, adc_columns, rsid_to_pos=None):
    
    adc_set = set(adc_columns)
    mapped = []

    for snp in adni_snps:
        adc_name = None

        if snp.startswith('rs'):
            if rsid_to_pos and snp in rsid_to_pos:
                pos = rsid_to_pos[snp]
                adc_name = f'chr19:{pos}'
        elif snp.startswith('19:'):
            adc_name = 'chr' + snp

        if adc_name and adc_name in adc_set:
            mapped.append(adc_name)

    return mapped


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
            print(f"  Fold {fold_i+1}: No features after variance filter, skipping")
            oof_rf[va_idx] = 0.5
            oof_tab[va_idx] = 0.5
            continue

        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_trn_fold, y_trn)
        oof_rf[va_idx] = rf.predict_proba(X_val_fold)[:, 1]

        
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
            oof_tab[va_idx] = tab.predict_proba(X_val_fold)[:, 1]
        except Exception as e:
            print(f"  Fold {fold_i+1}: TabNet error: {e}")
            oof_tab[va_idx] = 0.5

    
    X_meta = np.column_stack([oof_rf, oof_tab])
    meta_lr = LogisticRegression(random_state=42)
    meta_lr.fit(X_meta, y)
    oof_ensemble = meta_lr.predict_proba(X_meta)[:, 1]

    
    rf_weight = meta_lr.coef_[0][0]
    tab_weight = meta_lr.coef_[0][1]

    
    oof_simple_avg = (oof_rf + oof_tab) / 2

    
    results = {
        : roc_auc_score(y, oof_rf),
        : accuracy_score(y, (oof_rf >= 0.5).astype(int)),
        : roc_auc_score(y, oof_tab),
        : accuracy_score(y, (oof_tab >= 0.5).astype(int)),
        : roc_auc_score(y, oof_ensemble),
        : accuracy_score(y, (oof_ensemble >= 0.5).astype(int)),
        : roc_auc_score(y, oof_simple_avg),
        : float(rf_weight),
        : float(tab_weight),
    }

    return results


def main():
    print("=" * 60)
    print("External Validation on ADC NHW Cohort (DEDUPLICATED)")
    print("=" * 60)

    
    X_df, y = load_adc_data(ADC_RAW_FILE)

    
    rsid_to_pos = load_rsid_to_position_map()

    
    rankings = aggregate_snp_rankings(NESTED_CV_DIR, rsid_to_pos=rsid_to_pos)

    
    rankings = rankings.drop_duplicates(subset=['SNP'], keep='first')

    results_all = {}

    for top_n in TOP_N_LIST:
        print(f"\n{'='*60}")
        print(f"Evaluating Top {top_n} SNPs")
        print("=" * 60)

        
        top_snps_adni = rankings['SNP'].head(top_n).tolist()

        
        top_snps_adc = map_snp_names(top_snps_adni, X_df.columns.tolist(), rsid_to_pos)

        print(f"  ADNI top {top_n} SNPs -> {len(top_snps_adc)} found in ADC")

        if len(top_snps_adc) < 10:
            print(f"  WARNING: Only {len(top_snps_adc)} SNPs found, skipping")
            continue

        
        X_top = X_df[top_snps_adc].values
        X_top = np.nan_to_num(X_top, nan=0.0)

        print(f"  Running RF+TabNet 5-fold CV on {X_top.shape[0]} samples, {X_top.shape[1]} SNPs")

        
        results = run_rf_tabnet_cv(X_top, y, n_splits=N_CV_FOLDS)

        print(f"\n  Results for Top {top_n}:")
        print(f"    RF AUC: {results['rf_auc']:.4f}, ACC: {results['rf_acc']:.4f}")
        print(f"    TabNet AUC: {results['tabnet_auc']:.4f}, ACC: {results['tabnet_acc']:.4f}")
        print(f"    Ensemble (LR meta) AUC: {results['ensemble_auc']:.4f}, ACC: {results['ensemble_acc']:.4f}")
        print(f"    Simple Avg AUC: {results['simple_avg_auc']:.4f}")
        print(f"    Meta-learner coefs: RF={results['meta_rf_coef']:.3f}, TabNet={results['meta_tab_coef']:.3f}")

        results_all[f'top_{top_n}'] = {
            : top_n,
            : len(top_snps_adc),
            : len(y),
            : int(sum(y == 1)),
            : int(sum(y == 0)),
            **results
        }

    
    print(f"\n{'='*60}")
    print("Bottom SNP Controls")
    print("=" * 60)

    for top_n in TOP_N_LIST:
        
        bottom_snps_adni = rankings['SNP'].tail(top_n).tolist()
        bottom_snps_adc = map_snp_names(bottom_snps_adni, X_df.columns.tolist(), rsid_to_pos)

        if len(bottom_snps_adc) < 10:
            print(f"  Bottom {top_n}: Only {len(bottom_snps_adc)} SNPs found, skipping")
            continue

        X_bottom = X_df[bottom_snps_adc].values
        X_bottom = np.nan_to_num(X_bottom, nan=0.0)

        print(f"\n  Bottom {top_n}: {len(bottom_snps_adc)} SNPs found")

        results = run_rf_tabnet_cv(X_bottom, y, n_splits=N_CV_FOLDS)

        print(f"    RF AUC: {results['rf_auc']:.4f}")
        print(f"    TabNet AUC: {results['tabnet_auc']:.4f}")
        print(f"    Ensemble (LR meta) AUC: {results['ensemble_auc']:.4f}")
        print(f"    Meta-learner coefs: RF={results['meta_rf_coef']:.3f}, TabNet={results['meta_tab_coef']:.3f}")

        results_all[f'bottom_{top_n}'] = {
            : top_n,
            : len(bottom_snps_adc),
            **results
        }

    
    results_file = os.path.join(OUTPUT_DIR, "external_validation_results_dedup.json")
    with open(results_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    
    print("\n" + "=" * 60)
    print("EXTERNAL VALIDATION SUMMARY (DEDUPLICATED)")
    print("=" * 60)
    print(f"Dataset: ADC NHW (n={len(y)}, AD={sum(y==1)}, CN={sum(y==0)})")
    print("\nTop SNPs vs Bottom SNPs:")
    print("-" * 60)
    print(f"{'N':>6} | {'Top RF':>8} | {'Bot RF':>8} | {'Diff':>8} | {'Top Ens':>8} | {'Bot Ens':>8}")
    print("-" * 60)

    for top_n in TOP_N_LIST:
        top_key = f'top_{top_n}'
        bottom_key = f'bottom_{top_n}'
        if top_key in results_all and bottom_key in results_all:
            top_rf = results_all[top_key]['rf_auc']
            bottom_rf = results_all[bottom_key]['rf_auc']
            top_ens = results_all[top_key]['ensemble_auc']
            bottom_ens = results_all[bottom_key]['ensemble_auc']
            diff = top_rf - bottom_rf
            print(f"{top_n:>6} | {top_rf:>8.4f} | {bottom_rf:>8.4f} | {diff:>+8.4f} | {top_ens:>8.4f} | {bottom_ens:>8.4f}")


if __name__ == "__main__":
    main()
