# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:58:10 2022

@author: vaesc
"""


#   CONSTANTS

MAXFAN = 7 + 1  #value for fan, adjusted for range
MAXINV = 11 + 1 #value for inverter, adjusted for range
MAXTPHL = 1

#   VARIABLES

param = 0       #our first written parameter
minstep = 0     #best value of inverters
minfan = 0      #best value of fan

#   PROGRAM

################################################################################
# Project 4                                                                    #
# Steve Millman                                                                #
# Run hspice to determine the tphl of a circuit                                #
################################################################################

import numpy as np      # package needed to read the results file
import subprocess       # package needed to lauch hspice
import shutil           # package needed to copy files
import string           # package needed for character placement

from string import ascii_lowercase as asc 

################################################################################
# Start the main program here.                                                 #
################################################################################

# .param fan = 7
# Xinv1 a b inv M=1
# Xinv2 b c inv M=fan**1
# Xinv3 c d inv M=fan**2
# Xinv4 d e inv M=fan**3
# Xinv5 e f inv M=fan**4
# Xinv6 f g inv M=fan**5
# Xinv7 g h inv M=fan**6
# Xinv8 h i inv M=fan**7
# Xinv9 i j inv M=fan**8
# Xinv10 j k inv M=fan**9
# Xinv11 k z inv M=fan**10

MAXFAN = 7 + 1  #value for fan, adjusted for range
MAXINV = 11 + 1 #value for inverter, adjusted for range
mintphl = 100

#Loop through range 2 to 7
for step in range(1, MAXINV, 2):
    for fan in range(1, MAXFAN):
        #Loop through 1 - 11  for each inverter
        shutil.copy('esmeralda.sp', 'InvChain.sp')      #using the Esmeralda technique from lecture
        sp = open('InvChain.sp', "a")       #appending to our copied file
        param = '.param fan = ' + str(fan)      #first things first our param replacement
        sp.write('\n')          
        sp.write(str(param))        #writing param
        
        for inv in range(1, step+1):       #Loop for each individual node
            sp.write('\n')
            if (inv < step):        #while we havent reached the last node
                #write our inverter param lines to netlist
                write1 = 'Xinv' + str(inv) + ' ' + str(asc[inv-1]) + ' ' + str(asc[inv]) + ' ' + 'inv' + ' ' + 'M=fan**' + str(inv-1)
                sp.write(str(write1))
            elif (inv == step):
                #write our inverter param lines to netlist
                write2 = 'Xinv' + str(inv) + ' ' + str(asc[inv-1]) + ' ' + str(asc[25]) + ' ' + 'inv' + ' ' + 'M=fan**' + str(inv-1)
                sp.write(str(write2))
                sp.write('\n')
                sp.write('.end')        #need .end for netlist conclusion
                sp.close()      #close file
                
                # launch hspice. Note that both stdout and stderr are captured so
                # they do NOT go to the terminal!
                
                proc = subprocess.Popen(["hspice","InvChain.sp"],
                                        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                output, err = proc.communicate()
                
                # extract tphl from the output file
                data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
                tphl = data["tphl_inv"]
                print(tphl)
                print(fan)
                print(inv)
                
                #call our values for ideal inverter combination
                if (tphl < mintphl):
                    mintphl = tphl
                    minstep = step
                    minfan = fan
                        
#Output
sp = open('InvChain.sp', "r")

print('\n**************** Project 4 ******************')
print('\n', sp.read())
print('\nIdeal Fan/Inverter Combination: ')
print('\nTPHL: ', mintphl)
print('\n# of Inverters: ', minstep)
print('\n# of Fans: ', minfan)
print('\n')


                    
                    