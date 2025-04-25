#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

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
