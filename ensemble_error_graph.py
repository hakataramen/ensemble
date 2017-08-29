#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

error_range = np.arrange(0.0, 1.01, 0.01)
ens_erros = [ensemble_error(n_classifier=11, error=error)
	     for error in error_range]
plt.plot(error_range, ens_errors,
         label = 'Emsemble error',
         linewidth = 2)
plt.plot(error_range, error_range,
         linestyle='--', label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()





