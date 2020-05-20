#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   wave_function.py    
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/7/29 11:17 AM   hgh      1.0         None
import pywt


def waves(time_series, wave):
    ca, cd = pywt.dwt(time_series, wave)
    rec_ts = pywt.idwt(None, cd, wave)
    return rec_ts
