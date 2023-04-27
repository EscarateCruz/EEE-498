# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 00:23:37 2022

@author: vaesc
"""

import numpy as np                                     # needed for arrays
import matplotlib.pyplot as plt                        # matplot for plots
import scipy as sp
import pandas as pd                                    # needed to read the data

from scipy import optimize
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from warnings import filterwarnings 


# CONSTANTS
IS = 1e-9   #Saturation Current of Diode
N = 1.7     #Ideality Factor
R = 11000   #Resistance
T = 350     #Temperature
Q = 1.602e-19   #Charge Unit
K = 1.38e-23    #Boltzmanns Constant


# VARIABLES
vs = np.arange(0.1, 2.5, 0.1)   #Voltage Range from 0.1 to 2.5 in steps of 0.1
step = 24   #Step Size for Roots found by 2.5-0.1/0.1 
v = [1 for i in range(step)]        #Number of Step Sizes for Roots in fsolve


# PROGRAM

############################# PROBLEM 1 #######################################

#Diode Current Function
def Idiode(v, i, T, N):    
    return i * (np.exp((Q*v) / (N*K*T)) - 1)    #Using Equation Given in Stub

#Error Function
def error(vd, v, R, i, T, N):   
    err = (vd/R) - (v/R) + Idiode(vd, i, T, N)  #Using Equation Given in Stub
    return err

v_diode = fsolve(error, v, args = (vs, R, IS, N, T))    #Use fsolve to sweep voltage for roots
i_diode = Idiode(v_diode, IS, N, T)     #Use fsolve to sweep current for roots#Use fsolve to sweep voltage for roots
    
plt.plot(vs, np.log10(i_diode), label = "Source V vs. Diode I") #Source Voltage Data
plt.plot(v_diode, np.log10(i_diode), label = "Diode V vs. Diode I") #Diode Voltage Data
plt.title('Problem 1 Plot') #title
plt.xlabel('Source Voltage (V)')   #x/y labels
plt.ylabel('Diode Current (A)')  #x\y labels
plt.legend()    #legend to differentiate data
plt.show()



############################## PROBLEM 2 #######################################


################################################################################
# This function does the optimization for the resistor                         #
# Inputs:                                                                      #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

#Opening DiodeIV.txt
diode = np.loadtxt('DiodeIV.txt', dtype='float64')

#Variables
r_val = 10000                                         
ide_val = 1.5                                       
phi_val = 0.8                                              
area = 1e-8                                          
temp = 375                                                   
src_v = 0                            
meas_i = 0
P1_VDD_STEP = 0.6
x = 1
next_v = 0.1
errorA = 1
tol = 1e-9

src_v = diode[:, 0]     #Diode Voltage Array
meas_i = diode[:, 1]    #Diode Current Array

vdiode = np.zeros_like(src_v)   #useful for 


# Given current diode equation, removed Area from parameters to fit in Optimized stub function
def DiodeI(Vd, N, T, Is): 
    k = 1.380648e-23    #Boltzmann
    q = 1.6021766208e-19    #Charge unit constant
    Vt = N*k*T/q 
    #Is = A*T*T*np.exp(-phi*q/(k*T))     #current diode eq.
    return Is*(np.exp(Vd/Vt)-1) 

#Error Function, same as Problem 1 except using DiodeI not IDiode
def error2(vd, v, R, N, T, Is):   
    err = (vd/R) - (v/R) + DiodeI(vd, N, T, Is)  #Using Equation Given in Stub
    return err

#Optimized Resistance Function
def opt_r(r_value,ide_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage
    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(error2,prev_v,
(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis
    # compute the diode current
    diode_i = DiodeI(est_v,ide_value,temp,is_value)
    return meas_i - diode_i

#Optimized Ideality Function
def opt_n(ide_value,r_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage
    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(error2,prev_v,
(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis
    # compute the diode current
    diode_i = DiodeI(est_v,ide_value,temp,is_value)
    return (meas_i - diode_i) / (meas_i + diode_i + 1e-15)

#Optimized Phi Function
def opt_p(phi_value,r_value,ide_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage
    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(error2,prev_v,
(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis
    # compute the diode current
    diode_i = DiodeI(est_v,ide_value,temp,is_value)
    return (meas_i - diode_i) / (meas_i + diode_i + 1e-15)


################################################################################
# This is how leastsq calls opt_r                                              #
################################################################################

#iterate through optimizations to find our values, using a tolerance 1e-9, while average residual is greater

while (errorA > tol):
  
    #optimize R value
    r_val_opt = optimize.leastsq(opt_r,r_val,
                                 args=(ide_val,phi_val,area,temp,
                                       src_v,meas_i))
    r_val = r_val_opt[0][0]
    
    #optimize N value
    ide_val_opt = optimize.leastsq(opt_n,ide_val,
                                 args=(r_val,phi_val,area,temp,
                                       src_v,meas_i))
    ide_val = ide_val_opt[0][0]

    #optimize phi value
    phi_val_opt = optimize.leastsq(opt_p,phi_val,
                                args=(r_val,ide_val,area,temp,
                                      src_v,meas_i))
    phi_val = phi_val_opt[0][0]
    
    #phi = optimize.leastsq(phi_val, phi, all the other parameters including n and R) 
    #n = optimize.leastsq(ide_val, n, all the other parameters including n and R) 
    #R = optimize.leastsq(r_val, R, all the other parameters including n and R) 
    
    #finding average from sum and divison of residual, for matching our loop
    errorD = opt_p(phi_val,r_val,ide_val,area,temp,src_v,meas_i)
    esum = np.sum(np.abs(errorD))
    elen = len(errorD)
    errorA = esum/elen
    
    
    #Print our output per iteration
    print('\n')
    print('iteration: ', x)
    print('r_value: %.4f' %r_val)
    print('ide_value: %.4f' %ide_val)
    print('phi_value: %.4f' %phi_val)
    print('Error Average: %.9f' %errorA)
    
    x += 1  #counter/iteration number
   
#Final Values Output
print('\n')
print('---Final Values---')
print('r_value: %.4f' %r_val)
print('ide_value: %.4f' %ide_val)
print('phi_value: %.4f' %phi_val)

#Saturated Current Calculation
Is = area*temp*temp*np.exp(-phi_val*Q/(K*temp))

#use fsolve for finding diode voltage,
for i in range(len(src_v)):
    vd = optimize.fsolve(error2,next_v, (src_v[i], r_val, ide_val, temp, Is))   
    vdiode[i] = vd      # adding to array for later usage for diode current
    next_v += 0.1       #start estimate for roots of diode voltage
    
#final diode current calculation
Id = DiodeI(vdiode, ide_val, temp, Is)

#Plot
plt.plot(src_v, np.log10(meas_i), label = "Source V vs. Diode I", marker = 'o') #Source Voltage Data
plt.plot(src_v, np.log10(Id), label = "Diode V vs. Diode I") #Diode Voltage Data
plt.title('Problem 2 Plot') #title
plt.xlabel('Diode Voltage (V)')   #x/y labels
plt.ylabel('Diode Current (A)')  #x\y labels
plt.legend()    #legend to differentiate data
plt.show()
    
    
    