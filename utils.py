#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   utils.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2020/5/20 11:11 AM   hgh      1.0         None
import time

import numpy as np


def get_interval(ts_timestamp_array):
    timestamps = np.asarray(sorted(ts_timestamp_array))
    intervals = np.diff(timestamps)
    ts_interval = int(np.median(intervals))
    return ts_interval


def is_array(data):
    """
    判断数据是否是ndarray格式，如果不是则将数据转化为array类型
    :param data:
    :return: data numpy.array
    """
    if data is not np.ndarray:
        data = np.asarray(data)
    return data


def cal_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print("%s running time: %s sec." % (func.__name__, t2 - t1))
        return result

    return wrapper
