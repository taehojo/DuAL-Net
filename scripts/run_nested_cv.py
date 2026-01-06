#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DuAL-Net Nested Cross-Validation Script

Performs nested CV to avoid optimistic bias from alpha optimization.

Usage:
    python run_nested_cv.py <raw_file> <dx_file> <annotation_file> [options]

Options:
    --window_size: Window size for local analysis (default: 100)
    --top_n: Number of top SNPs to select (default: 100)
    --n_outer: Number of outer CV folds (default: 5)
    --n_inner: Number of inner CV folds (default: 5)
    --alpha_step: Alpha search step size (default: 0.1)
    --tabnet_epochs: TabNet training epochs (default: 50)
    --tabnet_patience: TabNet early stopping patience (default: 10)
    --output_dir: Output directory (default: result/nested-cv)
    --test_mode: Run with subset of data for quick testing
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import Logger
from src.nested_cv import nested_cv_evaluation
from src.annotation_transformer import transform_annotation_advanced

warnings.filterwarnings("ignore")


def load_data(raw_file, dx_file, annotation_file, logger, test_mode=False, test_n_samples=100):
    """Load and preprocess data files."""

    logger.log("Loading raw genotype data...")
    raw_df = pd.read_csv(raw_file, sep=r'\s+')

    # Drop non-SNP columns
    drop_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns], errors='ignore', inplace=True)

    # Clean column names (remove allele suffix)
    raw_df.columns = [c.split('_')[0] for c in raw_df.columns]

    logger.log(f"Loaded {raw_df.shape[0]} samples, {raw_df.shape[1]} SNPs")

    logger.log("Loading phenotype data...")
    dx_df = pd.read_csv(dx_file, sep=r'\s+')
    y = dx_df['New_Label'].values

    logger.log(f"Label distribution: 0={sum(y==0)}, 1={sum(y==1)}")

    logger.log("Loading annotation data...")
    anno_df = pd.read_csv(annotation_file)

    # Transform annotation if needed
    if 'rs_id' not in anno_df.columns:
        # Check for alternative column names
        for col in ['SNP', 'snp', 'rsid', 'variant_id']:
            if col in anno_df.columns:
                anno_df.rename(columns={col: 'rs_id'}, inplace=True)
                break

    logger.log(f"Loaded annotation for {len(anno_df)} SNPs")

    # Test mode: subset data
    if test_mode:
        logger.log(f"\n*** TEST MODE: Using {test_n_samples} samples ***")
        np.random.seed(42)

        # Stratified sampling
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]

        n_0 = min(test_n_samples // 2, len(idx_0))
        n_1 = min(test_n_samples - n_0, len(idx_1))

        selected_idx = np.concatenate([
            np.random.choice(idx_0, n_0, replace=False),
            np.random.choice(idx_1, n_1, replace=False)
        ])
        np.random.shuffle(selected_idx)

        raw_df = raw_df.iloc[selected_idx].reset_index(drop=True)
        y = y[selected_idx]

        logger.log(f"Test subset: {len(y)} samples (0={sum(y==0)}, 1={sum(y==1)})")

    return raw_df, y, anno_df


def main():
    parser = argparse.ArgumentParser(
        description='DuAL-Net Nested Cross-Validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('raw_file', help='Path to raw genotype file (.raw)')
    parser.add_argument('dx_file', help='Path to phenotype/diagnosis file')
    parser.add_argument('annotation_file', help='Path to SNP annotation file')

    parser.add_argument('--window_size', type=int, default=100,
                        help='Window size for local analysis')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top SNPs to select')
    parser.add_argument('--n_outer', type=int, default=5,
                        help='Number of outer CV folds')
    parser.add_argument('--n_inner', type=int, default=5,
                        help='Number of inner CV folds')
    parser.add_argument('--alpha_step', type=float, default=0.1,
                        help='Alpha search step size')
    parser.add_argument('--tabnet_epochs', type=int, default=50,
                        help='TabNet training epochs')
    parser.add_argument('--tabnet_patience', type=int, default=10,
                        help='TabNet early stopping patience')
    parser.add_argument('--output_dir', type=str, default='result/nested-cv',
                        help='Output directory for results')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run with subset of data for quick testing')
    parser.add_argument('--test_n_samples', type=int, default=100,
                        help='Number of samples to use in test mode')

    args = parser.parse_args()

    # Setup output directory and logger
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'nested_cv_logs.txt')
    logger = Logger(log_file=log_file)

    logger.log("=" * 60)
    logger.log("DuAL-Net Nested Cross-Validation")
    logger.log("=" * 60)
    logger.log(f"Raw file: {args.raw_file}")
    logger.log(f"DX file: {args.dx_file}")
    logger.log(f"Annotation file: {args.annotation_file}")
    logger.log(f"Output directory: {args.output_dir}")
    logger.log("")
    logger.log("Parameters:")
    logger.log(f"  Window size: {args.window_size}")
    logger.log(f"  Top N SNPs: {args.top_n}")
    logger.log(f"  Outer folds: {args.n_outer}")
    logger.log(f"  Inner folds: {args.n_inner}")
    logger.log(f"  Alpha step: {args.alpha_step}")
    logger.log(f"  TabNet epochs: {args.tabnet_epochs}")
    logger.log(f"  TabNet patience: {args.tabnet_patience}")
    if args.test_mode:
        logger.log(f"  TEST MODE: {args.test_n_samples} samples")
    logger.log("=" * 60)

    # Load data
    raw_df, y, anno_df = load_data(
        args.raw_file, args.dx_file, args.annotation_file,
        logger, test_mode=args.test_mode, test_n_samples=args.test_n_samples
    )

    # Run nested CV
    results = nested_cv_evaluation(
        raw_df=raw_df,
        y=y,
        anno_df=anno_df,
        logger=logger,
        n_outer_splits=args.n_outer,
        n_inner_splits=args.n_inner,
        window_size=args.window_size,
        top_n=args.top_n,
        alpha_step=args.alpha_step,
        tabnet_epochs=args.tabnet_epochs,
        tabnet_patience=args.tabnet_patience,
        result_dir=args.output_dir
    )

    # Print final summary
    logger.log("\n" + "=" * 60)
    logger.log("FINAL RESULTS")
    logger.log("=" * 60)
    summary = results['summary']
    logger.log(f"Test AUC: {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std']:.4f}")
    logger.log(f"  95% CI: [{summary['test_auc_ci_lower']:.4f}, {summary['test_auc_ci_upper']:.4f}]")
    logger.log(f"Test Accuracy: {summary['test_acc_mean']:.4f} +/- {summary['test_acc_std']:.4f}")
    logger.log(f"  95% CI: [{summary['test_acc_ci_lower']:.4f}, {summary['test_acc_ci_upper']:.4f}]")
    logger.log(f"Mean Best Alpha: {summary['mean_best_alpha']:.2f} +/- {summary['std_best_alpha']:.2f}")
    logger.log("=" * 60)

    # Save logs
    logger.write()
    logger.log(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
