# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:45:17 2020

@author: sandra_chang
"""

import keras
from keras import backend as K

print(keras.__version__)

import numpy 
print(id(numpy.dot) == id(numpy.core.multiarray.dot) )

print(K.floatx())