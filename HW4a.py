# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 01:55:13 2022

@author: vaesc
"""

#import re 
#str = '### Subject: scuba diving at... From: steve.millman@asu.edu Body: Underwater, where it is at' 
#match = re.search(r'[\w]*at$',str) 
#if match: 
#    print(match.group()) 
#else: 
#    print('did not find') 
    
    
import re 
str = '### <Subject: scuba diving at... From: steve.millman@asu.edu> Body: Underwater, where it is at> boi python rocks!' 
match = re.search(r'<.*?>', str) 
if match: 
    print(match.group()) 
else: 
    print('did not find') 

import re 
str = '### <Subject: scuba diving at... From: steve.millman@asu.edu> Body: Underwater, where it is at> boi python rocks!' 
match = re.search(r'<.*>', str) 
if match: 
    print(match.group()) 
else: 
    print('did not find') 
    
    
import re 
str = '### <Subject: scuba diving at... theuhhMOD0139X From: steve.millman@asu.edu Body: Underwater, where MoD 111220xdd20 is at> boi MOD20392049x' 
match = re.search(r'(?<=MOD)\d+(?=X)', str) 
if match: 
    print(match.group()) 
else: 
    print('did not find') 