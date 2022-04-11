# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:52:05 2022

@author: vaesc
"""

import numpy as np                                     # needed for arrays
import pandas as pd                                    # needed to read the data
import seaborn as sns                                  # data visualization
import matplotlib.pyplot as plt                        # matplot for plots

from sklearn import datasets                           # read the data sets
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # Perceptron algorithm
from sklearn.linear_model import LogisticRegression    # Logistic Regression algorithm
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.svm import SVC                            # Vector Machine algorithm
from sklearn.tree import DecisionTreeClassifier        # Tree algorithm
from sklearn.tree import export_graphviz               # a cool graph
from sklearn.ensemble import RandomForestClassifier    # Forest algorithm
from sklearn.neighbors import KNeighborsClassifier     # KNN algorithm
from sklearn.metrics import plot_roc_curve             #

#Constants

#Variables
x = 0
y = 0
c_val = 0


#Program

hearts = pd.read_csv('heart1.csv') 

#using access to list element
X = hearts[['age', 'sex', 'cpt', 'rbp', 'sc', 'fbs', 'rer', 'mhr', 'eia', 'opst', 'dests', 'nmvcf', 'thal']].values                    # separate the features we want
y = hearts.loc[:, 'a1p2'].values               # extract the values of heart disease classification 

# split the problem into train and test
# this will yield 70% training and 30% test
# random_state allows the split to be reproduced
# stratify=y not used in this case
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)

# scale X by removing the mean and setting the variance to 1 on all features.
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.
# (mean and standard deviation may be overridden with options...)
sc = StandardScaler()                    # create the standard scalar
sc.fit(X_train)                          # compute the required transformation
X_train_std = sc.transform(X_train)      # apply to the training data
X_test_std = sc.transform(X_test)        # and SAME transformation of test data



#PERCEPTRON ********************************************************
print('\n*********** PERCEPTRON ************')

# perceptron linear
# epoch is one forward and backward pass of all training samples
# (also known as an iteration)
# eta0 is rate of convergence
# max_iter, tol, if it is too low it is never achieved
# and continues to iterate to max_iter when above tol
# fit_intercept, fit the intercept or assume it is 0
# slowing it down is very effective, eta is the learning rate
ppn = Perceptron(max_iter=7, tol=1e-3, eta0=0.001,
             fit_intercept=True, random_state=0, verbose=True)
ppn.fit(X_train_std, y_train)              # do the training
print('Number in test ',len(y_test))
y_pred = ppn.predict(X_test_std)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f \n' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
# check results on combined data
y_combined_pred = ppn.predict(X_combined_std)
print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % \
      accuracy_score(y_combined, y_combined_pred))
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
 

 
#LOGISTIC REGRESSION *********************************************
print('\n*********** LOGISTIC REGRESSION ************')

c_val = 1

lr = LogisticRegression(C=c_val, solver='liblinear', \
                        multi_class='ovr', random_state=0) #perform algorithm
lr.fit(X_train_std, y_train)         # apply the algorithm to training data
y_pred = lr.predict(X_test)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f \n' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
# check results on combined data
y_combined_pred = lr.predict(X_combined_std)
print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % \
      accuracy_score(y_combined, y_combined_pred)) 

    

#SUPPORT VECTOR MACHINE *******************************************
print('\n*********** SUPPORT VECTOR MACHINE ************')
c_val = 0.1

svm = SVC(kernel='linear', C=c_val, random_state=0) #perform algorithm
svm.fit(X_train_std, y_train)                      # do the training
y_pred = svm.predict(X_test_std)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f \n' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
# check results on combined data
y_combined_pred = svm.predict(X_combined_std)
print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % \
      accuracy_score(y_combined, y_combined_pred))

    

#DECISION TREE LEARNING *******************************************
print('\n*********** DECISION TREE LEARNING ************')
# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy',max_depth=5 ,random_state=0) #perform algorithm
tree.fit(X_train,y_train) #do the training
y_pred = tree.predict(X_test)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f \n' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
# check results on combined data
y_combined_pred = tree.predict(X_combined)
print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % \
      accuracy_score(y_combined, y_combined_pred)) 


    
#RANDOM FOREST ****************************************************
print('\n*********** RANDOM FOREST ************')
trees = 101
forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, \
                                random_state=1, n_jobs=4)
forest.fit(X_train,y_train)

y_pred = forest.predict(X_test)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f \n' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
# check results on combined data
y_combined_pred = forest.predict(X_combined)
print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % \
      accuracy_score(y_combined, y_combined_pred)) 

    

#K-NEAREST NEIGHBOR ************************************************
print('\n*********** K-NEAREST NEIGHBOR ************')
neighs = 1  #best value of k
knn = KNeighborsClassifier(n_neighbors=neighs,p=2,metric='minkowski') #perform algorithm
knn.fit(X_train_std,y_train) #do the training

y_pred = knn.predict(X_test_std)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f \n' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))
# check results on combined data
y_combined_pred = knn.predict(X_combined_std)
print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % \
      accuracy_score(y_combined, y_combined_pred))
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')




