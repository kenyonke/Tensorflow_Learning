# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 02:15:24 2018

@author: kenyon
"""
import numpy as np
test_c = np.array([np.zeros((1,10)) for i in range(9)]).reshape((9,10))
print(test_c)
for i in range(9):
    test_c[i][i] = 1
print(test_c)