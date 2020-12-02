#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   wave_function.py    
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/7/29 11:17 AM   hgh      1.0         None
import pywt


def waves(time_series, wave):
    """
    提取小波变换特征
    :param time_series: 时间序列
    :param wave:
    :return:
    """
    ca, cd = pywt.dwt(time_series, wave)
    rec_ts = pywt.idwt(None, cd, wave)
    return rec_ts
