#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   tsad_features.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 9/28/20 4:27 PM   hgh      1.0         None
import sys
from Base import Base

if __name__ == '__main__':
    # 进行数据特征的保存
    data_path = sys.argv[1]
    save_path = "./result"
    base = Base()
    base.save_features(data_path=data_path, save_path=save_path)
