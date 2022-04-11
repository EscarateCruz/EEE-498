# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 23:52:30 2022

@author: vaesc
"""
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import quad
from matplotlib import pyplot as plt 


#Constants


#Variables
x = 0
z = 0
sol = 0

#Program
def fz(z):
    return z / (1-z) #SUBSTITUTION FUNCTION

def fx(x):
    return ((fz(x)/(1-x)**2)*np.exp(-fz(x)**2)) #New integral function w/sub

sol = quad(fx, 0, 1) #Evaluate New integral using 0 to 1 bounds

#Output
print('Value computed is {0:.8f} ' .format(sol[0])) #format and using first element
