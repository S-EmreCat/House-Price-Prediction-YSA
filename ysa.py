# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

# kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# veri yukleme
veriler = pd.read_csv('kc_house_data.csv')

veriler.info()

x_price=veriler.iloc[:,2:3]
y=veriler.iloc[:,3:]
X_price=x_price.values
Y=y.values

