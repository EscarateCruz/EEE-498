# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:22:48 2022

@author: vaesc
"""

import math                                # needed for arrays
import numpy as np   
import random as rand
import scipy as sp


from scipy.integrate import odeint
from warnings import filterwarnings                    # filtering warnings


#   CONSTANTS


#   VARIABLES
x = 0
y = 0
r = 0

pie = 0
square = 0
avg = 0


#   PROGRAM

o = [1, 2, 3, 4, 5, 6, 7]           #list of powers of ten to check

#iterate through our precisions
for j in o:
    p = []
    
    #iterate through our successes
    for i in range(100):
        circle = 0      #initial attempt values
        square = 1
        
        #using our 10000 tries from assignment
        while (square < 10000): 
            square += 1     #in the square (tries)            
            
            x = rand.uniform(0, 1)      #random number generator
            y = rand.uniform(0, 1)      #random number generator
            
            r = math.sqrt((x**2)+(y**2))    #calculating r from a^2+b^2=c^2
            
            if (r <= 1):        # if its under 1 its within the circle
                circle += 1   
            
            pie = circle / square       # equation for estimating pi/4
            py = np.pi/4            #actual pi/4 using numpy
            
            #check for numpy and calculated values of pi
            if (abs(pie - py) < (10**-j)/4):
                p.append(pie)       #add to the list if it matches the rounded j value\
                break
            
    #Formatting outputs depending on precision successes 
    if (len(p) == 0):
        avg = 'NO SUCCESS'     #get rid of 'nan'
    else:
        avg = (np.mean(p)) * 4       #finding average of the matched values
    
    
#   OUTPUT 

    print(10**-j, " success ",  len(p),  " times ", avg) 


    