'''
Created on Oct 15, 2015

@author: tim
'''
import numpy as np


with open('training_set.tsv','r') as f:
    train = f.readlines()[1:]
    
    
ids = []
qs = []
targets = []
choices = []
for q in train:
    
    line =  q.strip().split('\t')
    ids.append(line[0])
    qs.append(line[1])
    targets.append(line[2])
    choices.append(line[3:])

for q,a in zip(qs,choices):
    print q
    print a
    print '-'*100  
    

