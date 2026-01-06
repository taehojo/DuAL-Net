#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats

class Logger:
    def __init__(self, log_file=None):
        self.logs = []
        self.log_file = log_file

    def log(self, msg):
        print(msg)
        self.logs.append(str(msg))

    def write(self):
        if self.log_file:
            try:
                with open(self.log_file, 'w') as f:
                    for line in self.logs:
                        f.write(line + "\n")
                print(f"Logs saved -> {self.log_file}")
            except Exception as e:
                print(f"Error writing log file {self.log_file}: {e}")
        else:
            print("Log file not specified, logs not saved to file.")

def get_zero_var_idx(X):
    if isinstance(X, pd.DataFrame):
        var_ = X.var(axis=0)
        return X.columns[var_ == 0].tolist()
    elif isinstance(X, np.ndarray):
        var_ = X.var(axis=0)
        return np.where(var_ == 0)[0]
    else:
        raise TypeError("Input X must be a pandas DataFrame or numpy ndarray.")

def remove_zv_by_idx(X, zero_idx):
    if len(zero_idx) > 0:
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=zero_idx)
        elif isinstance(X, np.ndarray):
            return np.delete(X, zero_idx, axis=1)
    return X

def split_into_windows_as_sequence(data, window_size):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    arr = data.values
    N, M = arr.shape
    W = M // window_size
    if W == 0:
        print(f"Warning: Number of features ({M}) is less than window size ({window_size}). No windows created.")
        return np.empty((N, 0, window_size))
    arr = arr[:, :W * window_size]
    seqs = []
    for i in range(W):
        st = i * window_size
        en = st + window_size
        seqs.append(arr[:, st:en])
    if not seqs:
        return np.empty((N, 0, window_size))
    return np.stack(seqs, axis=1)


def calculate_95_ci(values):
    """
    Calculate 95% Confidence Interval using t-distribution.

    Args:
        values: list or array of fold-wise metrics (e.g., AUCs)

    Returns:
        dict with keys: mean, std, se, ci_lower, ci_upper, n
        Returns NaN values if insufficient data
    """
    # Remove NaN values
    values = [v for v in values if not np.isnan(v)]
    n = len(values)

    result = {
        'n': n,
        'mean': np.nan,
        'std': np.nan,
        'se': np.nan,
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        't_critical': np.nan
    }

    if n == 0:
        return result

    result['mean'] = np.mean(values)

    if n < 2:
        return result

    result['std'] = np.std(values, ddof=1)  # sample std
    result['se'] = result['std'] / np.sqrt(n)
    result['t_critical'] = stats.t.ppf(0.975, df=n-1)

    result['ci_lower'] = result['mean'] - result['t_critical'] * result['se']
    result['ci_upper'] = result['mean'] + result['t_critical'] * result['se']

    return result


def format_ci_string(ci_result, decimal=4):
    """
    Format 95% CI result as a string for reporting.

    Args:
        ci_result: dict from calculate_95_ci()
        decimal: number of decimal places (default: 4)

    Returns:
        str: formatted string like "0.6781 (95% CI: 0.6234-0.7328)"
    """
    if np.isnan(ci_result['mean']):
        return "N/A"

    mean_str = f"{ci_result['mean']:.{decimal}f}"

    if np.isnan(ci_result['ci_lower']) or np.isnan(ci_result['ci_upper']):
        return f"{mean_str} (95% CI: N/A)"

    return f"{mean_str} (95% CI: {ci_result['ci_lower']:.{decimal}f}-{ci_result['ci_upper']:.{decimal}f})"
