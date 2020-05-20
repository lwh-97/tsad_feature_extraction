#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   ts_preprocess.py
# @Contact :   15271221@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2019/7/29 11:21 AM   hgh      1.0         None

import pandas as pd
import numpy as np

from utils import get_interval, cal_time


class GetData:

    def __init__(self, data_path):
        self.data_path = data_path
        self.dataFrame = None
        self.ts_data_array = None
        self.ts_timestamp_array = None
        self.ts_label_array = None
        self.interval = None

    def _fine_tune_timestamp(self):
        """
        调整航信数据的数据时间戳是13位的情况，去除多余的后三位，基本都是000

        """
        self.dataFrame["timestamp"] = self.dataFrame.iloc[:, 0].astype(str).str[:10]
        self.dataFrame["timestamp"] = self.dataFrame["timestamp"].astype(int)

    def _endian_format_uniformity(self):
        """
        解决航信数据时间戳尾数不一致的问题

        """
        self.dataFrame["timestamp"] = (self.dataFrame["timestamp"] // 10) * 10

    def _sort_values(self):
        """
        解决航信数据时间戳顺序不规整，使其按时间戳从小到大排序

        """
        self.dataFrame = self.dataFrame.sort_values("timestamp")
        self.dataFrame = self.dataFrame.reset_index(drop=True)

    def _fill_nan(self):
        """
        解决label数据中出现缺失值的现象
        :return:
        """
        self.dataFrame = self.dataFrame.fillna(int(0))
        self.dataFrame["label"] = self.dataFrame["label"].astype(int)

    def _drop_duplicates(self):
        """
        去除数据中出现的重复数据，保留重复数据中的第一条

        """
        self.dataFrame.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        # 这个重新设置index的操作一定不能少，否则导致后续对dataFrame的操作有影响
        self.dataFrame = self.dataFrame.reset_index(drop=True)

    def _drop_incorrect_intervals(self):
        """
        去除航信数据中时间间隔不为1分钟的数据点

        """

        ts_timestamp_array = self.dataFrame["timestamp"].values
        diffs = np.diff(ts_timestamp_array)
        # TODO 后续优化可把 ！= 60 改为 % 60 ！= 0，以及如何适应间隔不是60的情况
        self.interval = get_interval(ts_timestamp_array)
        diffs_where = np.where(diffs != self.interval)[0]
        remove_list = []
        for i in diffs_where:
            if i not in remove_list:
                start_index = i
                end_index = i + 1
                if end_index == (len(self.dataFrame) - 1):
                    remove_list.append(end_index)
                    break
                while ((ts_timestamp_array[end_index] -
                       ts_timestamp_array[end_index - 1]) != self.interval):
                    remove_list.append(end_index)
                    end_index = end_index + 1
                gap = ts_timestamp_array[end_index] - ts_timestamp_array[start_index]
                add_all = (self.interval - gap % self.interval) % self.interval
                self.dataFrame.loc[end_index:, ["timestamp"]] += add_all
        self.dataFrame = self.dataFrame.drop(remove_list)
        self.dataFrame = self.dataFrame.reset_index(drop=True)

    # TODO 装饰器后续移除，仅为测试
    @cal_time
    def get_data_from_file(self):
        """
        extract the time series set from given file.

        """
        try:
            self.dataFrame = pd.read_csv(self.data_path, engine='python')
            self.dataFrame = self.dataFrame.reset_index(drop=True)
            if 'timestamp' in self.dataFrame.columns and "value" in self.dataFrame.columns:
                pass
            elif 'ds' in self.dataFrame.columns and 'y' in self.dataFrame.columns:
                self.dataFrame.columns = ['timestamp', 'value']
            else:
                raise Exception
        except IOError:
            # TODO 可以根据不同的错误，报不同的错误信息，比如传入的不是csv文件
            print("读取csv文件出错: " + self.data_path)
            return
        self._sort_values()
        if "label" in self.dataFrame.columns:
            self._fill_nan()
        self._fine_tune_timestamp()
        self._endian_format_uniformity()
        self._drop_duplicates()
        self._drop_incorrect_intervals()

        self.ts_data_array = np.asarray(self.dataFrame["value"])
        self.ts_timestamp_array = np.asarray(self.dataFrame["timestamp"])
        if "label" in self.dataFrame.columns:
            self.ts_label_array = np.asarray(self.dataFrame["label"])
