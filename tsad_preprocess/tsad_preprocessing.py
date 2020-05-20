#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   tsad_preprocessing.py    
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/8/20 10:24 PM   hgh      1.0         None
from tsad_preprocess.ts_deal_missing_data import DealMissingData
from tsad_preprocess.ts_extraction import GetData
from tsad_preprocess.ts_scale import get_scaled_time_series


def pre_processing(data_path, deal_missing=True, scaled=False, scaled_kind=None):
    # 从文件中获取训练数据
    get_data = GetData(data_path)
    get_data.get_data_from_file()

    ts_data_array = get_data.ts_data_array
    ts_timestamp_array = get_data.ts_timestamp_array
    ts_label_array = get_data.ts_label_array
    ts_interval = get_data.interval

    # 对训练数据进行缺失值处理
    if deal_missing is True:
        deal_missing_data = DealMissingData(ts_data_array, ts_timestamp_array, ts_label_array)
        deal_missing_data.deal_missing_data()

        # 获取缺失值处理后的训练数据
        ts_data_array = deal_missing_data.ts_data_array
        ts_timestamp_array = deal_missing_data.ts_timestamp_array
        ts_label_array = deal_missing_data.ts_label_array

    # 对训练数据的time series进行min-max缩放
    if scaled is True:
        ts_data_array = get_scaled_time_series(ts_data_array, scale_function=scaled_kind)

    return ts_data_array, ts_timestamp_array, ts_label_array, ts_interval
