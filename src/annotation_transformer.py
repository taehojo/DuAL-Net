#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def transform_annotation_advanced(df, logger):
    if not isinstance(df, pd.DataFrame):
        logger.log("Error: Input to transform_annotation_advanced must be a DataFrame.")
        return pd.DataFrame()

    df_ = df.copy()
    logger.log("Starting annotation transformation...")

    drop_cols = ['chrom', 'pos', 'allele1', 'allele2', 'conservation_score', 'gene_name']
    cols_to_drop = [c for c in drop_cols if c in df_.columns]
    if cols_to_drop:
        df_.drop(columns=cols_to_drop, inplace=True)
        logger.log(f"Dropped columns: {cols_to_drop}")

    if 'exon_intron_utr' in df_.columns:
        def parse_region(t):
            if isinstance(t, str):
                try:
                    if t.startswith('(') and t.endswith(')') or t == 'None':
                         t = eval(t)
                    else:
                        return "none"
                except:
                    return "none"

            if not isinstance(t, tuple) or len(t) != 3:
                return "none"
            a, b, c = t
            a, b, c = bool(a), bool(b), bool(c)

            if a and not b and not c:
                return "exon"
            elif not a and b and not c:
                return "intron"
            elif not a and not b and c:
                return "utr"
            elif a and c:
                return "exon"
            return "none"

        df_['region'] = df_['exon_intron_utr'].apply(parse_region)
        df_.drop(columns=['exon_intron_utr'], inplace=True)
        logger.log("Processed 'exon_intron_utr' into 'region'.")
    else:
        logger.log("Column 'exon_intron_utr' not found. Creating 'region' column with 'none'.")
        df_['region'] = "none"

    def is_nonempty_list(x):
        if isinstance(x, str):
            if x.startswith('[') and x.endswith(']'):
                try:
                    x = eval(x)
                except:
                     return 0
            else:
                 return 0

        if not isinstance(x, list):
            return 0
        return 1 if len(x) > 0 else 0

    list_features = ['regulatory_features', 'epigenetic_features']
    for feat in list_features:
        col_name = f"has_{feat.split('_')[0]}"
        if feat in df_.columns:
            df_[col_name] = df_[feat].apply(is_nonempty_list)
            df_.drop(columns=[feat], inplace=True)
            logger.log(f"Processed '{feat}' into '{col_name}'.")
        else:
            logger.log(f"Column '{feat}' not found. Creating '{col_name}' column with 0.")
            df_[col_name] = 0

    clin_cols = [
        'uncertain_significance', 'likely_pathogenic', 'pathogenic', 'drug_response', 'other',
        'risk_factor', 'association', 'protective', 'established_risk_allele'
    ]
    for c in clin_cols:
        if c in df_.columns:
            df_[c] = df_[c].fillna(0).astype(int)
            logger.log(f"Processed clinical column: '{c}'.")
        else:
            logger.log(f"Clinical column '{c}' not found. Creating it with 0.")
            df_[c] = 0

    categorical_cols = ['most_severe_consequence', 'gene_biotype']
    for c in categorical_cols:
        if c in df_.columns:
            df_[c] = df_[c].fillna('unknown').astype(str)
            logger.log(f"Processed categorical column: '{c}'.")
        else:
            logger.log(f"Categorical column '{c}' not found. Creating it with 'unknown'.")
            df_[c] = 'unknown'

    ohe_cols = ['region'] + categorical_cols
    ohe_cols_present = [c for c in ohe_cols if c in df_.columns]
    if ohe_cols_present:
        df_ = pd.get_dummies(df_, columns=ohe_cols_present, dummy_na=False)
        logger.log(f"One-hot encoded columns: {ohe_cols_present}")
    else:
        logger.log("No columns found for one-hot encoding.")

    logger.log("Annotation transformation finished.")
    return df_
