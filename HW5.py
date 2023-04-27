# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:12:03 2022

@author: vaesc
"""

import numpy as np                                     # needed for arrays
import matplotlib.pyplot as plt                        # matplot for plots
import scipy as sp

from scipy.integrate import odeint
from warnings import filterwarnings 


#   CONSTANTS
RANGE = np.linspace(0, 7, 700)      #Time Steps provided in PDF


#   VARIABLES
Result = 0


#   PROGRAM

###############################################################################    
# PROBLEM 1 
###############################################################################

# Cosine Function of Time
def P1(y, t):
    y = np.cos(t)
    return y

#Odeint Function to 700 Point Range
P1R = odeint(P1, 1, RANGE)

# Plot
plt.plot(RANGE, P1R, marker = 'o') #Problem 1 Output Data w/ Range
plt.title('Problem 1: y = cos(t)') #title
plt.xlabel('Time (t)')   #x/y labels
plt.ylabel('Result (y)')  #x\y labels
plt.legend()    #legend to differentiate data
plt.show()


###############################################################################    
# PROBLEM 2
###############################################################################

# Differential #1 Function of Time
def P2(y, t):
    dy = (-y) + ((t**2) * (np.exp(-2*t))) + 10
    return dy

#Odeint Function to 700 Point Range
P2R = odeint(P2, 0, RANGE)

# Plot
plt.plot(RANGE, P2R, marker = 'o') #Problem 2 Output Data w/ Range
plt.title('Problem 2: dy = -y + t2e-2t + 10') #title
plt.xlabel('Time (t)')   #x/y labels
plt.ylabel('Result (y)')  #x\y labels
plt.legend()    #legend to differentiate data
plt.show()


###############################################################################    
# PROBLEM 3
###############################################################################

# Differential #2 Function of Time
def P3(y, t):
    ddy = (25*np.cos(t)) + (25*np.sin(t)) - (4*y) - (4*y) 
    return ddy

#y = [1, 1]

#Odeint Function to 700 Point Range
P3R = odeint(P3, 1, RANGE)  

# Plot
plt.plot(RANGE, P3R, marker = 'o') #Problem 3 Output Data w/ Range
plt.title('Problem 3: ddy = 25cos(t) + 25sin(t) - 4y - 4dy') #title
plt.xlabel('Time (t)')   #x/y labels
plt.ylabel('Result (y)')  #x\y labels
plt.legend()    #legend to differentiate data
plt.show()





