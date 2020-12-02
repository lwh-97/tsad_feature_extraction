#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   ts_scale.py    
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/7/29 2:22 PM   hgh      1.0         None


def get_scaled_time_series(ts_data_array, scale_function=None):
    """
    获取放缩后的时序
    :param ts_data_array: 待检测时间序列
    :param scale_function:
    :return:
    """
    if scale_function is not None:
        print("making" + str(scale_function))
        ts_data_array = scale_function(ts_data_array)
    return ts_data_array
