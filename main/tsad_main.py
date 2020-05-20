#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @File    :   tsad_main.py    
# @Contact :   19120364@bjtu.edu.com

# @Modify Time      @Author    @Version    @Description
# ------------      -------    --------    -----------
# 2020/5/20 11:38 AM   hgh      1.0         None
from Base import Base

if __name__ == '__main__':
    base = Base()
    # 进行数据预处理
    base.pre_processing()
    # 进行训练
    base.fit()
    # 进行预测
    base.predict()
