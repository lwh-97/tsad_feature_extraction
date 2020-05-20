#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   statistic_function.py    
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/7/29 11:14 AM   hgh      1.0         None
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt


def get_feature_logs(time_series):
    return np.log(time_series + 1e-2)


# def get_feature_SARIMA_residuals(time_series):
#     predict = SARIMAX(time_series,
#                       trend='n',
#                       order=(5, 1, 1),
#                       measurement_error=True).fit().get_prediction()
#     return time_series - predict.predicted_mean


def get_feature_SimpleES_residuals(time_series, alpha):
    predict = SimpleExpSmoothing(time_series).fit(smoothing_level=alpha)
    return time_series - predict.fittedvalues


def get_feature_AddES_residuals(time_series, alpha, beta):
    predict = ExponentialSmoothing(time_series, trend='add').fit(smoothing_level=alpha, smoothing_slope=beta)
    return time_series - predict.fittedvalues


def get_feature_Holt_residuals(time_series, alpha, beta, phi):
    predict = Holt(time_series).fit(smoothing_level=alpha, smoothing_slope=beta, damping_slope=phi)
    return time_series - predict.fittedvalues


if __name__ == '__main__':
    # test = np.asarray(np.random.rand(100))
    test = np.array([1, 1, 2, 3, 4])
    print(np.log(-1))
