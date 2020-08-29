# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:43:42 2020

@author: sandra_chang
"""

import numpy as np
from numpy import *
import matplotlib.pylab as plt


#Sigmoid 數學函數表示方式
#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#Sigmoid 微分
def dsigmoid(x):
    return (x * (1 - x))

def softmax(x):
     return np.exp(x) / float(sum(np.exp(x)))
 
def ReLU(x):
     return abs(x)*(x>0)

def dReLU(x):
     return 1*(x>0)

def plotImage(method):
    
    plt.figure()
    # linespace generate an array from start and stop value
    # with requested number of elements. Example 10 elements or 100 elements.
    x = plt.linspace(-10,10,100)
    
    if (method == "sigmoid"):
        plt.plot(x, sigmoid(x), 'b', label='linspace(-10,10,10)')
    elif (method == "dsigmoid"):
        plt.plot(x, dsigmoid(x), 'b', label='linspace(-10,10,10)')
    elif (method == "softmax"):
        plt.plot(x, softmax(x), 'b', label='linspace(-10,10,10)')
    elif (method == "ReLU"):
        plt.plot(x, ReLU(x), 'b')
    elif (method == "dReLU"):
        plt.plot(x, dReLU(x), 'b')
    else:
        return
    
    # Draw the grid line in background.
    plt.grid()
    
    # 顯現圖示的Title
    plt.title(method)
    
    # 顯現 the Sigmoid formula
#    plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)
    
#    #resize the X and Y axes
#    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
#    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
     
    # create the graph
    plt.show()
    
methods = {"sigmoid","dsigmoid","softmax","ReLU","dReLU"}

for method in methods:
    plotImage(method)