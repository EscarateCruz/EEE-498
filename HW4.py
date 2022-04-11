# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:19:46 2022

@author: vaesc
"""

import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg          

#Constants

FN_MEAN_RETURN     = 'Mean Return (%): '
FN_STDEV_RETURN   = 'Standard Deviation Return (%): '
FN_YRCONTRIBUTION    = 'Yearly Contribution: '
FN_NUM_YRCONTRIBUTION = '# of Years of Contribution: '
FN_NUM_RETIREMENT  = '# of Years to Retirement: '
FN_SPEND    = 'Annual Retirement Spend: '

FIELD_NAMES = [ FN_MEAN_RETURN, FN_STDEV_RETURN, FN_YRCONTRIBUTION, FN_NUM_YRCONTRIBUTION, FN_NUM_RETIREMENT, FN_SPEND ] #array to set our indeces
NUM_FIELDS = 6    # how many fields there are

B_CALCULATE = 'Calculate'    # need things in more than one place
B_QUIT    = 'Quit'


#Variables

mean_return = 0  
stdev_return = 0
yrcontribution = 0
num_yrcontribution = 0 
num_retirement = 0
spend = 0

noise = 0
wi = 0
wealth = 0
lastyear = 0
wealth_at_retirement = 0

wealthplot = np.zeros((70, 10), dtype = float) #creating a matrix (70 years by 10 runs of wealth analysis) will easily store all our data


#Program

plt.figure()        #used m4_pysimplegui.py as template


def calculate(window,entries):
    
    #get our entries and assign them to variables
    mean_return = float(entries[FN_MEAN_RETURN])  
    stdev_return = float(entries[FN_STDEV_RETURN])  
    yrcontribution = float(entries[FN_YRCONTRIBUTION]) 
    num_yrcontribution = float(entries[FN_NUM_YRCONTRIBUTION]) 
    num_retirement = float(entries[FN_NUM_RETIREMENT])  
    spend = float(entries[FN_SPEND])
    
    wealth = np.sum(wealthplot[(int(num_retirement) +1), :], axis = 0)  #grab the last year of retirement, and get the sum at that year (1st row)
    
    for x in range(10):    #doing 10 analysis runs in our wealth calc.
        wi = 0
        last_yr = 0
        noise = (stdev_return/100)*np.random.randn(70) # noise equation 
        
        for i in range(70):    #running through the 70 years and doing our calculations
            if (wi >= 0):   #first make sure we have money (or at least starting out with nothing, no negatives aka in debt)
                wealthplot[i, x] = wi   #add our current wealth number to our ith year and xth analysis in the matrix data saver
                last_yr = i + 1 #not counting the "0th year"
            else:
                break   #if it's negative, then tough luck
            #now to store and add to our Wi wealth variable
            if ((num_yrcontribution - 1) > i): #From Start year i until Contributions End
                wi = wi * (1 + (mean_return/100) + noise[i]) + yrcontribution #EQUATION 1 
            elif ((num_yrcontribution - 1) > i): #From End of Contributions until Retirement
                    wi = wi * (1 + (mean_return/100) + noise[i]) #EQUATION 2
            else: #From Retirement to the end
                wi = wi * (1 + (mean_return/100) + noise[i]) - spend #EQUATION 3
        
        #setting up plot devices
        plt.plot(range(last_yr), wealthplot[0 : last_yr, x], '-x') #plot for x axis our years 0 to 70, y our wealth index 
        plt.title('Wealth Over 70 Years')
        plt.xlabel('Years')
        plt.ylabel('Wealth')  
    plt.show()
    
    return (wealth / 10) #return the wealth average (from our analyseses)


# set the font and size
sg.set_options(font=('Helvetica',20))
# The layout is a list of lists
# Each each entry in the top-level list is a row in the GUI
# Each entry in the next-level lists is a widget in that row
# Order is top to bottom, then left to right

layout = []                                 # start with the empty list
for index in range(NUM_FIELDS):             # for each of the fields to create
    layout.append([sg.Text(FIELD_NAMES[index]+': ',size=(20,1)), \
                   sg.InputText(key=FIELD_NAMES[index],size=(10,1))])   # adding our field names and values places to the GUI layout
layout.append([sg.Button(B_CALCULATE), sg.Button(B_QUIT)])      #add our buttons calc and quit to layout
layout.append([sg.Text(wealth_at_retirement, key = 'RESULT', size = (30, 1))])      
    

# Output

# start the window manager
window = sg.Window('Homework 4: 70 Year Wealth Calculator', layout)         #window header
# Run the event loop
while (True):
    event, values = window.read()
    #print(event,values)
    if (event == sg.WIN_CLOSED) or (event == B_QUIT):   #quit the program upon command
        break
    if (event == B_CALCULATE):      #do what the program is supposed to do
        wealth_at_retirement = (calculate(window, values))      #perform our function
        window['RESULT'].update(str('Wealth at Retirement: $ {:,}'.format(int(wealth_at_retirement))))       #using the key in our layout and wealth variable we update our monetary wealth value in the GUI box
        
window.close()

