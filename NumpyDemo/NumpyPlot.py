#!/usr/bin/env python  
# encoding: utf-8  
"""
@author: Alfons
@contact: alfons_xh@163.com
@file: NumpyPlot.py 
@time: 2018/1/19 14:51 
@version: v1.0 
"""
import numpy as np
import matplotlib.pyplot as plt

print("-" * 40 + "2D图形" + "-" * 40)
x = 1
y = 1
plt.plot(x, y, 'o')
plt.show()

x = np.linspace(-1, 1, 1000, endpoint = True)
y = np.cos(x)
plt.plot(x, y, 'r--')
plt.show()

