# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import csv

#randomly generate 6 groups of numbers
SS = np.random.randint(low=0,high=21,size=4)
SD = np.random.randint(low=0,high=6,size=4)
BR = np.random.randint(low=0,high=4,size=4)
IRI = np.random.randint(low=0,high=45,size=4)
A = np.random.randint(low=7,high=57,size=4)
PD = np.random.randint(low=7,high=57,size=4)
#unchanged x,y
X = ([0.37,0.4,0.4,0.37])
Y = ([0.52,0.45,0.55,0.48])
#combined into an array
N = np.vstack([X,Y,SS,SD,BR,IRI,A,PD])
N = N.T
agent = (["subj's family"],["subj's family"],["subj's family"],["subj's family"])
N = np.hstack([agent,N])
#generate into csv
with open ('S004b.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['agent','x','y','social support','social distance','biological relative','IRI','Affiliation','Power Distance'])
    writer.writerows(N)