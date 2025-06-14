import numpy as np
import scipy.signal as sig
from functions import utils

def apply_sav_gol(signal_row, sg_win):
    smooth_cal = sig.savgol_filter(signal_row, window_length=sg_win, deriv=0, delta=1, polyorder=3)
    smooth_deriv = sig.savgol_filter(signal_row, window_length=sg_win, deriv=1, delta=1, polyorder=3)
    return smooth_cal, smooth_deriv

def calculate_mask(spikes_row, win_len, num_cols):
    mask = np.ones(num_cols, dtype=bool)
    event_spikes = np.where(spikes_row)[0]
    for i in event_spikes:
        start = max(0, i - win_len)
        end = min(num_cols, i + win_len)
        mask[start:end] = False
    return mask

def perform_polyfit(smooth_cal, smooth_deriv, mask):
    valid_indices = np.where(mask)[0]
    if valid_indices.size > 1:
        b = np.polyfit(smooth_cal[valid_indices], smooth_deriv[valid_indices], 1)[0]
        return b
    else:
        return 0

def calculate_feed(smooth_cal, smooth_deriv, b):
    return (-b) * smooth_cal + smooth_deriv

def dask_preprocess(signal, spikes, sg_win=31, win_len=5):
    if signal.shape[1] != spikes.shape[1]:
        spikes = spikes[:, -signal.shape[1]:]

    num_rows, num_cols = signal.shape
    feed = np.zeros((num_rows, num_cols))
    b_fits = np.zeros(num_rows)

    for row in range(num_rows):
        smooth_cal, smooth_deriv = apply_sav_gol(signal[row], sg_win)
        mask = calculate_mask(spikes[row], win_len, num_cols)
        b_fits[row] = perform_polyfit(smooth_cal, smooth_deriv, mask)
        feed[row, :] = calculate_feed(smooth_cal, smooth_deriv, b_fits[row])

    return feed

def dask_estimate_kernels(signal, spikes, sg_win=31, win_len=5):
    if signal.shape[1] != spikes.shape[1]:
        spikes = spikes[:, -signal.shape[1]:]

    num_rows, num_cols = signal.shape
    b_fits = np.zeros(num_rows)

    for row in range(num_rows):
        smooth_cal, smooth_deriv = apply_sav_gol(signal[row], sg_win)
        mask = calculate_mask(spikes[row], win_len, num_cols)
        b_fits[row] = perform_polyfit(smooth_cal, smooth_deriv, mask)

    return b_fits
