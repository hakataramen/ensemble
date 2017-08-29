#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np

a = np.argmax(np.bincount([0, 0, 1], weights = [0.2, 0.2, 0.6]))
print(a)
