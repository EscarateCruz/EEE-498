# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:38:15 2022

@author: vaesc
"""

import numpy as np                                     # needed for arrays
import pandas as pd                                    # needed to read the data
import matplotlib.pyplot as plt                        # matplot for plots

from sklearn import datasets                           # read the data sets
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.decomposition import PCA                  # PCA package
from sklearn.metrics import confusion_matrix           # generate the  matrix
from sklearn.neural_network import MLPClassifier       # MLP algorithm from neural network
from warnings import filterwarnings 

#Constants

#Variables

X = 0
y = 0
i = 0
features = 0
test_accuracy = []
list_accuracy = []
components = []


#Program

filterwarnings('ignore') 

sonar = pd.read_csv('sonar_all_data_2.csv')     #first things first, data reading

features = sonar.shape[1] - 2         #number of features from length of CSV, minus rock/mine classification

#print(features)
#using access to list element ()
X = sonar.iloc[:, 0 : features - 1].values       # separate the features we want
y = sonar.iloc[:, features].values               # extract the values  

# split the problem into train and test
# this will yield 70% training and 30% test
# random_state allows the split to be reproduced
# stratify=y not used in this case
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=1)

#rocks = 0                # initialize counters
#mines = 0
#for obj in y_test:    # for all of the objects in the test set
#    if obj == 2:         # mines are class 2, rocks are class 1
#        mines += 1     # increment the appropriate counter
#    else:
#        rocks += 1
#print("rocks",rocks,"   mines",mines)    # print the results

sc = StandardScaler()                    # create the standard scalar
sc.fit(X_train)                          # compute the required transformation
X_train_std = sc.transform(X_train)      # apply to the training data
X_test_std = sc.transform(X_test)        # and SAME transformation of test data


#Loop used to go through accuracy of each component in features list
for i in range(1, features):
    components.append(i)                        #used to display component # being tested
    print("Testing Component #:", i)

    pca = PCA(n_components=i, random_state=1)   #PCA algorithm usage
    
    X_train_pca = pca.fit_transform(X_train)    # apply to the train data
    X_test_pca = pca.transform(X_test)          # do the same to the test data
    
    model = MLPClassifier( hidden_layer_sizes=(100), activation='logistic', max_iter=2000, alpha=0.00001,  solver='adam', tol=0.0001, random_state=1 )  #using MLP algorithm
    
    X_train_mlp = model.fit(X_train_pca, y_train)              # do the training
    y_pred = model.predict(X_test_pca)              # how do we do on the test data?

    print('*********************')
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    accuracy = accuracy_score(y_test, y_pred) * 100   # get the accuracy of component i and round it for integer
    print('Accuracy: %.2f \n' %round(accuracy, 3))
    test_accuracy.append(accuracy)      #add to the test array
max_accuracy = max(test_accuracy)       #find the maximum accuracy score 
max_index = test_accuracy.index(max_accuracy)       #find max index from the maximum accuracy score


#PART 2: ACCURACY AND COMPONENTS RESULTS

list_accuracy = np.vstack((components, test_accuracy)).T
print('\n\n 2) RESULTS: ')
out = pd.DataFrame(list_accuracy, columns=['Component #:', 'Accuracy:'])    #use DataFrame from Pandas to organize our output results of components and accuracy scores
print('\n', out)


#PART 3: MAX ACCURACY, NUMBER OF COMPONENTS

print('\n\n 3) Maximum Accuracy, Number of Components')
print('\n Maximum Accuracy: ', max_accuracy)
print('\n in Number of Components: ', components[max_index-1])      #we want to find the components using max accuracy, so we find it at max index


#PART 4: PLOT

plt.plot(components, test_accuracy) #data
plt.title('\n\n 4) Accuracies vs. # of Components') #title
plt.xlabel('\n Number of Components (#)')   #x/y labels
plt.ylabel('\n Accuracies (%)')
plt.show()


#PART 5: CONFUSION MATRIX

print('\n\n 5) Confusion Matrix')
cmat = confusion_matrix(y_test, y_pred)     #using confusion_matrix algorithm from sklearn
print('\n', cmat)

           



