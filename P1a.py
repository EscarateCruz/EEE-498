# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:51:31 2022

@author: vaesc
"""

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import seaborn as sns # data visualization
import matplotlib.pyplot as plt

#Constants

#Variables
corr = 0
cov = 0
with_corr = 0
with_cov = 0

#Program

hearts = pd.read_csv('heart1.csv') 

print("Find NULL values...")
print(hearts.isnull(),"\n")                      # T if entry is missing or NaN
print("Find NULL values and sum them...")
print(hearts.isnull().sum(),"\n")

# read in the data
# create the correlation
# take the absolute value since large negative are as useful as large positive
corr = hearts.corr().abs()

# set the correlations on the diagonal or lower triangle to zero,
# so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)
# We clear the diagonal since the correlation with itself is always 1.
# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.
# Note this will be element by element multiplication
corr *= np.tri(*corr.values.shape, k=-1).T
print(corr)

# now unstack it so we can sort things
# note that zeros indicate no correlation OR that we cleared below the
# diagonal. Note that corr_unstack is a pandas series.
corr_unstack = corr.unstack()
print(corr_unstack)
print(type(corr_unstack))
print('\n')


# Sort values in descending order
corr_unstack.sort_values(inplace=True,ascending=False)
print(corr_unstack)
print('\n')

# Now just print the top values
print(corr_unstack.head(5))
print('\n')

# Get the correlations with type
with_corr = corr_unstack.get(key="a1p2")
print('Correlation\n', with_corr)
print('\n')

# create the covariance
# take the absolute value since large negative are as useful as large positive
cov = hearts.cov().abs()

# set the covariance on the diagonal or lower triangle to zero,
# so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)
# We clear the diagonal since the covariance with itself is always 1.
# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.

# Note this will be element by element multiplication
cov *= np.tri(*cov.values.shape, k=-1).T
print(cov)
print('\n')

# now unstack it so we can sort things
# note that zeros indicate no covariance OR that we cleared below the
# diagonal. Note that cov_unstack is a pandas series.
cov_unstack = cov.unstack()
print(cov_unstack)
print(type(cov_unstack))
print('\n')

# Sort values in descending order
cov_unstack.sort_values(inplace=True,ascending=False)

# Now just print the top values
print(cov_unstack.head(5))
print('\n')

# Get the covariance with type
with_cov = cov_unstack.get(key="a1p2")
print('Covariance\n', with_cov)
print('\n')

#Pair Plot
sns.set(style='whitegrid', context='notebook')   # set the apearance
sns.pairplot(hearts,height=1.5)                    # create the pair plots
plt.show() 


