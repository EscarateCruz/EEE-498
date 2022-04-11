# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 23:49:50 2022

@author: vaesc
"""

import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy import integrate
from matplotlib import pyplot as plt 

# Constants
A = 0
B = 0
C = 0

# Variables
x = 0
y = 0
r = 0
fx = 0
out = 0
sol = []

# Program
def integral(A, B, C):
    y = [i*0.01 for i in range(500)] #0.01 steps 0 to 5
    sol = [] #for solution
    
    def fx(x):
        return (A*x*x) + (B*x) + C #Function given to integrate
    
    for r in y: #Going through the integral bound (0 to 5)
        out = scipy.integrate.quad(fx, 0, r) #integrate at current step
        out = out[0] #ignore error 
        sol.append(out) #add to list
        
    plt.plot(y, sol) #plot the list of integrated values

integral(2, 3, 4)
integral(2, 1, 1)


# Plot 
plt.xlabel("R Values")  
plt.ylabel("Integral Values")  

plt.title("HW2: Integral for A")  
plt.legend(["F(X) for 2, 3, 4", "F(X) for 2, 1, 1"])  
plt.show()  
