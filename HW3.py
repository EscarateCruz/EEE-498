# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:31:55 2022

@author: vaesc
"""

#Constants


#Variables
maxnode = 0
maxvoltage = 0
error = 0

node_cnt = 0
volt_cnt = 0
sum_cnt = 0

voltages = []
currents = []

#Program

################################################################################
# Created on Fri Aug 24 13:36:53 2018                                          #
#                                                                              #
# @author: olhartin@asu.edu; updates by sdm                                    #
#                                                                              #
# Program to solve resister network with voltage and/or current sources        #
################################################################################

import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants
# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]


################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################


def get_dimensions(netlist):           # pass in the netlist
    ### EXTRA STUFF HERE!
    maxnode = 0
    maxvoltage = 0
    
    for i in range(len(netlist)): #going through the netlist 'array'
        if (netlist[i][0] == 1): #
            maxvoltage = maxvoltage + 1 #voltage count
            
        if ((netlist[i][2] or netlist[i][3]) > maxnode): #check if our element is greater than current max
            if ((netlist[i][2]) == (netlist[i][3])):  
                error = 1 #extraneous instance
            elif ((netlist[i][2]) > (netlist[i][3])):
                maxnode = netlist[i][2] #if the 2 - R value is greater than 3, use 2
            elif ((netlist[i][2]) < (netlist[i][3])):
                maxnode = netlist[i][3] #if the 3 - R value is greater than 2, use 3
    
    #convert the counts to integers 
    node_cnt = int(maxnode)
    volt_cnt = int(maxvoltage)
    
    print('Nodes: ', node_cnt, '\n', 'Voltage sources: ', volt_cnt, '\n')
    
    return node_cnt, volt_cnt


################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
################################################################################



def stamper(y_add,netlist,currents,num_nodes):
    # return the total number of rows in the matrix for
    # error checking purposes
    # add 1 for each voltage source...
    for comp in netlist:                  # for each component...
       ############### print(' comp ', comp)            # which one are we handling...
        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix
        
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1
        
        if ( comp[COMP.TYPE] == COMP.R ):           # a resistor
            if (i >= 0):                            # add on the diagonal
                y_add[i,i] += 1.0/comp[COMP.VAL]    # take component value and add to diagnol
            if (j >= 0):                            # add on the diagonal
                y_add[j,j] += 1.0/comp[COMP.VAL]    # take component value and add to diagnol
            if (i >= 0) and (j >= 0):                            # add on the diagonal
                y_add[i,j] += -1.0/comp[COMP.VAL]               #INVERSE DIAGNOL TEST
                y_add[j,i] += -1.0/comp[COMP.VAL]

        elif ( comp[COMP.TYPE] == COMP.IS ):           # a current source          
                if (i >= 0):                            
                    if (comp[COMP.VAL] >= 0):
                        currents[i] = currents[i] - 1.0*comp[COMP.VAL] # take component value and add to diagnol
                    else:
                        currents[i] = currents[i] + 1.0*comp[COMP.VAL] # take component value and add to diagnol
                if (j >= 0):                            # make sure
                    if (comp[COMP.VAL] >= 0):
                        currents[j] = currents[j] + 1.0*comp[COMP.VAL] # take component value and add to diagnol
                    else:
                        currents[j] = currents[j] - 1.0*comp[COMP.VAL] # take component value and add to diagnol
                    
        elif ( comp[COMP.TYPE] == COMP.VS ):           # a voltage source
            num_nodes = num_nodes + 1   #for keeping add to voltage source count
            if (i >= 0):                          
                    y_add[num_nodes - 1, i] = 1.0 # take component value and add to diagnol
                    y_add[i, num_nodes - 1] = 1.0 # take component value and add to diagnol
                    
            if (j >= 0):                          
                    y_add[num_nodes - 1, j] = -1.0 # take component value and add to diagnol
                    y_add[j, num_nodes - 1] = -1.0 # take component value and add to diagnol
                    
            currents[num_nodes - 1] = comp[COMP.VAL] #Component values are loaded
            voltages[num_nodes - 1] = 0 #location = 0 for voltage
                    
                    
            #EXTRA STUFF HERE!
    node_cnt = num_nodes 
    return node_cnt  # should be same as number of rows!


################################################################################
# Start the main program now...                                                #
################################################################################
# Read the netlist!


netlist = read_netlist()

print("\n")

# Print the netlist so we can verify we've read it correctly
for i in range(len(netlist)):
    print(netlist[i])
print("\n")

#EXTRA STUFF HERE!

node_cnt, volt_cnt = get_dimensions(netlist) #get our required dimensions/size
print('count: ', node_cnt + volt_cnt)
sum_cnt = node_cnt + volt_cnt #add the counts for use in the zeros function pre-package
y_add = np.zeros((sum_cnt, sum_cnt), dtype = float)

#create matrices (columns in our case) for the current and voltage analysis
currents = np.zeros(sum_cnt, dtype = float)
voltages = np.zeros(sum_cnt, dtype = float)

node_cnt = stamper(y_add, netlist, currents, node_cnt) #matrix creation with correct dimensions

voltages = np.linalg.solve(y_add, currents)


#Output
if (error == 1):
    print('Netlist Unaccepted; ')
    

print("\n")


#the following for loops are engineered to output the matrix column
#as it is: a column. For aesthetic purposes.

print('Currents: ')
for x in range(len(currents)):
    if(currents[x] < 0):
        print('[', np.round(currents[x], 1), ']')
    else:
        print('[ ', np.round(currents[x], 1), ']')
print("\n")
#print(currents)

print('Voltages: ')
for x in range(len(voltages)):
    if(voltages[x] < 0):
        print('[', np.round(voltages[x], 1), ']')
    else:
        print('[ ', np.round(voltages[x], 1), ']')
print("\n")
#print(voltages)





