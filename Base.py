#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   Base.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2020/5/20 11:22 AM   hgh      1.0         None
import configparser
import datetime
import os
import numpy as np

from sklearn.metrics import precision_recall_curve

from tsad_feature.microsoft_spectral_residual_function import get_spectral_residual_time_series
from tsad_feature.statistic_function import get_feature_SimpleES_residuals, get_feature_AddES_residuals, \
    get_feature_Holt_residuals, get_feature_logs
from tsad_feature.wave_function import waves
from tsad_period_algorithm import InterpACF
from tsad_preprocess.tsad_preprocessing import pre_processing
from utils import is_array
from sklearn.ensemble import RandomForestClassifier


class Base:
    def __init__(self):
        dir_path = os.path.abspath(os.path.dirname(__file__))
        self.conf = configparser.ConfigParser()
        self.config_path = dir_path + "/properties.ini"
        self.conf = configparser.ConfigParser()
        self.conf.read(self.config_path, encoding='utf-8')
        # 数据文件地址
        self.data_path = self.conf.get("path", "data_path")
        self.result_dir = self.conf.get("path", "result_dir")
        self.win_array = eval(self.conf.get("feature_param", "windows_array"))
        self.ts_data_array = None
        self.ts_timestamp_array = None
        self.ts_label_array = None
        self.ts_interval = None
        # 开始时间
        self.start_time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

    def get_ts_data_array(self):
        """
        获取时间序列数据
        :return:
        """
        return self.ts_data_array.copy()

    def set_ts_data_array(self, ts_data_array):
        """
        设置时间序列数据
        :param ts_data_array:
        :return:
        """
        ts_data_array = is_array(ts_data_array)
        self.ts_data_array = ts_data_array.copy()

    def get_ts_timestamp_array(self):
        """
        获取时间序列时间戳
        :return:
        """
        return self.ts_timestamp_array.copy()

    def set_ts_timestamp_array(self, ts_timestamp_array):
        """
        设置时间序列时间戳
        :param ts_timestamp_array:
        :return:
        """
        ts_timestamp_array = is_array(ts_timestamp_array)
        self.ts_timestamp_array = ts_timestamp_array.copy()

    def get_ts_label_array(self):
        """
        获取时间序列标签
        :return:
        """
        if self.ts_label_array is not None:
            return self.ts_label_array.copy()
        else:
            return None

    def set_ts_label_array(self, ts_label_array):
        """
        设置时间序列的标签
        :param ts_label_array:
        @return:
        """
        if ts_label_array is None:
            self.ts_label_array = None
        else:
            ts_label_array = is_array(ts_label_array)
            self.ts_label_array = ts_label_array.copy()

    def pre_processing(self):
        """
        数据预处理
        :return:
        """
        # 获取数据
        print('Parsing Data')
        scaled = eval(self.conf.get("pre_processing", "scaled"))
        # python 反射机制
        sklearn_pre = __import__("sklearn.preprocessing", fromlist=[""])
        scaled_kind = getattr(sklearn_pre, self.conf.get("pre_processing", "scaled_kind"))
        self.ts_data_array, self.ts_timestamp_array, self.ts_label_array, self.ts_interval = \
            pre_processing(self.data_path, deal_missing=True, scaled=scaled,
                           scaled_kind=scaled_kind)

    def _decompose_warmup_point(self):
        """
        获取每条时序的周期大小
        :return: 预热窗口大小和数据周期
        """
        period_detect = InterpACF()
        period = period_detect.detect(self.ts_data_array, self.ts_timestamp_array)
        print("周期为：" + str(period))
        return period + 100, period

    def _get_feature(self):
        """
        提取数据特征
        :return:
        """
        self.ts_feature_data = []
        self.ts_feature_data_label = []
        start_point, period = self._decompose_warmup_point()

        start_accum = 0

        # features of spectral residual
        spectural_residual = get_spectral_residual_time_series(self.ts_data_array.copy())

        # features of SimpleES
        time_series_simple_es_1 = get_feature_SimpleES_residuals(self.ts_data_array, 0.1)
        time_series_simple_es_3 = get_feature_SimpleES_residuals(self.ts_data_array, 0.3)
        time_series_simple_es_5 = get_feature_SimpleES_residuals(self.ts_data_array, 0.5)
        time_series_simple_es_7 = get_feature_SimpleES_residuals(self.ts_data_array, 0.7)
        time_series_simple_es_9 = get_feature_SimpleES_residuals(self.ts_data_array, 0.9)

        # features of AddES
        time_series_addes_24 = get_feature_AddES_residuals(self.ts_data_array, 0.2, 0.4)
        time_series_addes_46 = get_feature_AddES_residuals(self.ts_data_array, 0.4, 0.6)
        time_series_addes_64 = get_feature_AddES_residuals(self.ts_data_array, 0.6, 0.4)
        time_series_addes_82 = get_feature_AddES_residuals(self.ts_data_array, 0.8, 0.2)

        # feature of Holt
        time_series_holt_246 = get_feature_Holt_residuals(self.ts_data_array, 0.2, 0.4, 0.6)
        time_series_holt_468 = get_feature_Holt_residuals(self.ts_data_array, 0.4, 0.6, 0.8)
        time_series_holt_684 = get_feature_Holt_residuals(self.ts_data_array, 0.6, 0.8, 0.4)
        time_series_holt_842 = get_feature_Holt_residuals(self.ts_data_array, 0.8, 0.4, 0.2)

        # feature of waves : db4, coif4, sym8, dmey, rbio2.8, haar
        rec_ts_db4 = waves(self.ts_data_array, 'db4')
        rec_ts_coif4 = waves(self.ts_data_array, 'coif4')
        rec_ts_sym8 = waves(self.ts_data_array, 'sym8')
        rec_ts_dmey = waves(self.ts_data_array, 'dmey')
        rec_ts_rbio28 = waves(self.ts_data_array, 'rbio2.8')
        rec_ts_haar = waves(self.ts_data_array, 'haar')

        # features from tsa models for time series logarithm
        time_series_logs = get_feature_logs(self.ts_data_array)
        columns_name = []
        for i in np.arange(start_point, len(self.ts_data_array)):
            columns_name = []
            # the datum contains features of each points in a ts
            datum = []
            datum_label = self.ts_label_array[i]

            diff_plain = np.abs(self.ts_data_array[i] - self.ts_data_array[i - 1])
            columns_name.append("abs_diff_plain")
            diff_plain_percent = diff_plain / (self.ts_data_array[i - 1] + 1e-10)
            columns_name.append("diff_plain_percentage")
            datum.append(diff_plain_percent)
            datum.append(diff_plain)

            diff_day = np.abs(self.ts_data_array[i] - self.ts_data_array[i - period])
            columns_name.append("abs_diff_day")
            datum.append(diff_day)
            diff_day_percent = diff_day / (self.ts_data_array[i - period] + 1e-10)
            columns_name.append("diff_day_percentage")
            datum.append(diff_day_percent)

            start_accum = start_accum + self.ts_data_array[i]
            mean_accum = start_accum / (i - start_point + 1)
            columns_name.append("mean_accum")
            datum.append(mean_accum)

            columns_name.append("logs")
            datum.append(time_series_logs[i])

            columns_name.append("abs_diff of accumulated mean and current value")
            datum.append(np.abs(self.ts_data_array[i] - mean_accum))

            columns_name.append('spectral residual')
            datum.append(spectural_residual[i])

            columns_name.append('SimpleES1')
            datum.append(time_series_simple_es_1[i])
            columns_name.append('SimpleES3')
            datum.append(time_series_simple_es_3[i])
            columns_name.append('SimpleES5')
            datum.append(time_series_simple_es_5[i])
            columns_name.append('SimpleES7')
            datum.append(time_series_simple_es_7[i])
            columns_name.append('SimpleES9')
            datum.append(time_series_simple_es_9[i])

            columns_name.append('AddES24')
            datum.append(time_series_addes_24[i])
            columns_name.append('AddES46')
            datum.append(time_series_addes_46[i])
            columns_name.append('AddES64')
            datum.append(time_series_addes_64[i])
            columns_name.append('AddES82')
            datum.append(time_series_addes_82[i])

            columns_name.append('Holt246')
            datum.append(time_series_holt_246[i])
            columns_name.append('Holt468')
            datum.append(time_series_holt_468[i])
            columns_name.append('Holt684')
            datum.append(time_series_holt_684[i])
            columns_name.append('Holt842')
            datum.append(time_series_holt_842[i])

            columns_name.append('db4')
            datum.append(rec_ts_db4[i])
            columns_name.append('coif4')
            datum.append(rec_ts_coif4[i])
            columns_name.append('sym8')
            datum.append(rec_ts_sym8[i])
            columns_name.append('dmey')
            datum.append(rec_ts_dmey[i])
            columns_name.append('rbio28')
            datum.append(rec_ts_rbio28[i])
            columns_name.append('haar')
            datum.append(rec_ts_haar[i])

            for k in self.win_array:
                mean_w = np.mean(self.ts_data_array[i - k:i + 1])
                columns_name.append("win_mean_" + str(k))
                datum.append(mean_w)
                var_w = np.mean((np.asarray(self.ts_data_array[i - k:i + 1]) - mean_w) ** 2)
                columns_name.append("win_var_" + str(k))
                datum.append(var_w)

                mean_w_and_1 = mean_w + (self.ts_data_array[i - k - 1] - self.ts_data_array[i]) / (k + 1)
                var_w_and_1 = np.mean((np.asarray(self.ts_data_array[i - k - 1:i]) - mean_w_and_1) ** 2)
                win_diff_mean = mean_w - mean_w_and_1
                columns_name.append("win_diff_mean_" + str(k))
                datum.append(win_diff_mean)
                win_diff_var = var_w - var_w_and_1
                columns_name.append("win_diff_var_" + str(k))
                datum.append(win_diff_var)

                columns_name.append("abs_current time - mean_" + str(k))
                datum.append(np.abs(self.ts_data_array[i] - mean_w))

            self.ts_feature_data.append(np.asarray(datum))
            self.ts_feature_data_label.append(np.asarray(datum_label))

        self.columns_name = columns_name
        print("提取的特征：" + str(self.columns_name))

    def _fit_init(self):
        """

        :return:
        """
        # 开始训练模型
        self.classifier_name = "RandomForestClassifier"
        self.param_grid = {"criterion": 'gini',
                           "class_weight": None,
                           "max_features": 0.035,
                           "min_samples_split": 2,
                           'min_samples_leaf': 2,
                           'max_depth': 35,
                           'n_estimators': 60
                           }
        print("分类器模型是：" + str(self.classifier_name))
        self.classifier = RandomForestClassifier(**self.param_grid)

    def fit(self):
        """

        :return:
        """
        self._fit_init()
        if self.ts_data_array is not None and self.ts_label_array is not None:
            self._get_feature()
            self.classifier.fit(self.ts_feature_data, self.ts_feature_data_label)
            print("fit finish")
        else:
            print("please pre_processing first")
            return

    @staticmethod
    def _get_best_f1_score(fpr, tpr, threshold):
        """
        获取best f1 score
        :param fpr:
        :param tpr:
        :param threshold: 阈值
        :return:
        """
        f1s = 2 * fpr * tpr / (fpr + tpr)
        max_args = np.argmax(f1s)
        return f1s[max_args], fpr[max_args], tpr[max_args], threshold[max_args]

    def predict(self):
        """
        进行预测
        """
        # 在训练集上看效果
        with open(self.result_dir + '/' + str(self.classifier_name) + '_train_set_performance.txt', 'a') as f:
            f.write(self.start_time)
            proba = self.classifier.predict_proba(self.ts_feature_data)
            fpr, tpr, thre = precision_recall_curve(self.ts_feature_data_label, proba[:, 1])
            maxs, argmax_fpr, argmax_tpr, argmax_thre = self._get_best_f1_score(fpr, tpr, thre)
            f.write("the data's best f1_score: %f, fpr: %f, tpr: %f, thre: %f \n" %
                       (maxs, argmax_fpr, argmax_tpr, argmax_thre))
        print("predict finish")

