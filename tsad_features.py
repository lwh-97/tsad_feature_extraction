#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   tsad_features.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 9/28/20 4:27 PM   hgh      1.0         None
from Base import Base

if __name__ == '__main__':
    base = Base()
    # 进行数据的保存
    data_path = "../data/1th_ts_train.csv"
    base.save_features(data_path=data_path)
