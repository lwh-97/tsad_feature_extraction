#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   microsoft_spectral_residual_function.py    
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/8/20 9:31 PM   hgh      1.0         None
import numpy as np

MAX_RATIO = 0.25
EPS = 1e-8
THRESHOLD = 3
MAG_WINDOW = 3
SCORE_WINDOW = 100


def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res


def spectral_residual_transform(values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """

    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS

    mag_log = np.log(mag)
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - average_filter(mag_log, n=MAG_WINDOW))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag


def predict_next(values):
    """
    Predicts the next value by sum up the slope of the last value with previous values.
    Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
    where g(x_i,x_j) = (x_i - x_j) / (i - j)
    :param values: list.
        a list of float numbers.
    :return : float.
        the predicted next value.
    """

    if len(values) <= 1:
        raise ValueError('data should contain at least 2 numbers')

    v_last = values[-1]
    n = len(values)

    slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

    return values[1] + sum(slopes)


def extend_series(values, extend_num=5, look_ahead=5):
    """
    extend the array data by the predicted next value
    :param values: list.
        a list of float numbers.
    :param extend_num: int, default 5.
        number of values added to the back of data.
    :param look_ahead: int, default 5.
        number of previous values used in prediction.
    :return: list.
        The result array.
    """

    if look_ahead < 1:
        raise ValueError('look_ahead must be at least 1')

    extension = [predict_next(values[-look_ahead - 2:-1])] * extend_num
    return values + extension


def get_spectral_residual_time_series(ts_data):
    print("extracting spectral residual feature")
    if ts_data is not list:
        try:
            ts_data = ts_data.tolist()
        except AttributeError:
            print("请输入list或拥有tolist属性的数据类型")

    extended_series = extend_series(ts_data)
    mag = spectral_residual_transform(extended_series)[:len(ts_data)]
    ave_mag = average_filter(mag, n=SCORE_WINDOW)
    ave_mag[np.where(ave_mag <= EPS)] = EPS

    return abs(mag - ave_mag) / ave_mag
    # return spectral_residual_transform(ts_data)

