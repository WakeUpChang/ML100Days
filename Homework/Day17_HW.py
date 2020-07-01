# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:14:11 2020

@author: sandra_chang
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

# 初始設定 Ages 的資料
ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})

# 新增欄位 "equal_width_age", 對年齡做等寬劃分
ages["equal_width_age"] = pd.cut(ages["age"], 4)

# 觀察等寬劃分下, 每個種組距各出現幾次
print(ages["equal_width_age"].value_counts()) # 每個 bin 的值的範圍大小都是一樣的

# 新增欄位 "equal_freq_age", 對年齡做等頻劃分
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)

# 觀察等頻劃分下, 每個種組距各出現幾次
print(ages["equal_freq_age"].value_counts()) # 每個 bin 的資料筆數是一樣的


ages["customized_age_grp"] = pd.cut(ages["age"],bins= [0,10,20,30,50,100])

print(ages["customized_age_grp"].value_counts())