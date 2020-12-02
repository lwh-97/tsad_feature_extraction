#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   ts_deal_missing_data.py    
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/7/29 2:28 PM   hgh      1.0         None
import numpy as np
import pandas as pd

from utils import get_interval


class DealMissingData:
    def __init__(self, ts_data_array, ts_timestamp_array, ts_label_array=None):
        """

        :param ts_data_array: 待检测时间序列
        :param ts_timestamp_array:  时间戳
        :param ts_label_array:  数据标签
        """
        self.ts_data_array = ts_data_array
        self.ts_timestamp_array = ts_timestamp_array
        self.ts_label_array = ts_label_array
        self.ts_interval = None
        self.ts_num = None
        self.full_data_array = None
        self.full_timestamp_array = None
        self.full_label_array = None

    def _get_interval(self):
        """
        获取数据采样间隔
        @return:
        """
        self.ts_interval = get_interval(self.ts_timestamp_array)
        self.ts_num = np.ceil((self.ts_timestamp_array[-1] - self.ts_timestamp_array[0]) / self.ts_interval) + 1
        print('''the time series in file : 所给数据时间区间内应有数据点个数 = %d, 
                                           实际数据点个数 = %d, 
                                           缺失个数 = %d, 
                                           数据采样间隔 = %d seconds'''
              % (int(self.ts_num), len(self.ts_timestamp_array),
                 int(self.ts_num)-len(self.ts_timestamp_array), self.ts_interval))

    def deal_missing_data(self, interpolation_method='linear'):
        """
        处理缺失数据
        :param interpolation_method: 插值方法
        :return: 插值后的数据
        """
        self._get_interval()
        ts_index = np.arange(self.ts_timestamp_array[0],
                             self.ts_timestamp_array[-1] + self.ts_interval,
                             self.ts_interval)
        full_zero_series = pd.Series(np.zeros(int(self.ts_num), dtype=np.int64), index=ts_index)
        # ts_data_series 中不存在的，相加就会变成nan,后续使用插值填补
        ts_data_series = pd.Series(self.ts_data_array, index=self.ts_timestamp_array)
        full_data_series = ts_data_series + full_zero_series
        full_data_series = full_data_series.interpolate(method=interpolation_method)
        self.full_data_array = full_data_series.values
        if self.ts_label_array is not None:
            # 同理，对标签也需要填补，将nan后续填补为0，即正常值
            ts_label_series = pd.Series(self.ts_label_array, index=self.ts_timestamp_array)
            full_label_series = ts_label_series + full_zero_series
            full_label_series = full_label_series.fillna(0)
            self.full_label_array = full_label_series.values
        self.full_timestamp_array = ts_index
