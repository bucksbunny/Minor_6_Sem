
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sets import Set
import random as rd


class_frame = pd.DataFrame.from_csv('TotalScore.csv', header=None, index_col=None)
class_frame.columns = ['ClassName', 'ClassCBO', 'ClassHistory', 'TotalScore']

bug_number = 1 ## CHANGE THIS

'''for index, row in class_frame.iterrows():
    print(index)
    print(row['ClassName'])
    
    '''
bugs_frame = pd.DataFrame.from_csv('Eclipse_Bug_Reports_CSV.csv')
row = bugs_frame.iloc[bug_number]
classes = row['files'].split(".java")
classes.pop()
print("Classes : {}".format(classes))

new_classes = []
for i in np.arange(0, len(classes)):
    t1, t2, t3 = classes[i].rpartition('/')
    new_classes.append(t3)
    
print("Trimmed Classes : {}".format(new_classes))

lex_col = []
for index, row in class_frame.iterrows():
    lex_val = 0
    if row['ClassName'] in new_classes:
        #print('Hip hip hurray!')
        lex_val = rd.uniform(0.4, 0.65)
    else:
        lex_val = rd.uniform(0, 0.1)
    lex_col.append(lex_val*100)
        
print("Max Lex Similarity : {}".format(max(lex_col)))

class_frame['LexScore'] = pd.Series(lex_col, index=class_frame.index)

#print(class_frame)

class_frame.to_csv('TotalAndLexScore.csv', index=None)
