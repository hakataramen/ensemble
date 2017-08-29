#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])

p = np.average(ex, axis=0, weights=[0.2 ,0.2, 0.6])

print("p", p)
print("np.argmax(p)",np.argmax(p))
