#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   interp_acf.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2020/4/20 1:01 PM   hgh      1.0         None
from interpacf import interpolated_acf, dominant_period

import numpy as np

from utils import is_array, get_interval


class InterpACF:
    """
    qfxiao调研并提供本时序周期性检测算法

    """
    period_name = "InterpACF"

    @staticmethod
    def detect(ts_data, ts_timestamp):
        ts_data = is_array(ts_data)
        ts_timestamp = is_array(ts_timestamp)
        ts_data_ = ts_data - np.mean(ts_data)
        lag, acf = interpolated_acf(ts_timestamp, ts_data_)
        # TODO: 当使用csv_test.csv时，出现detected_period为non的情况
        detected_period = dominant_period(lag, acf, plot=False)
        interval = int(get_interval(ts_timestamp))
        return int(detected_period // interval)
