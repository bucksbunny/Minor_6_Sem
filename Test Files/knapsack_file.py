
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sets import Set
import random as rd


# In[27]:

class_frame = pd.DataFrame.from_csv('TotalAndLexScore.csv', header=0, index_col=None)

knapsack_capacity = 10 ## CHANGE THIS

#print(class_frame)

strheader = "knapsack problem specification (1 knapsacks, "+str(len(class_frame))+" items)\n=\n"
strheader = strheader+"knapsack 1:\n capacity: +"+str(knapsack_capacity)

print(class_frame.columns)

for index, row in class_frame.iterrows():
    profit_1 = row['TotalScore']
    profit_2 = row['LexScore']
    profit = (0.2)*profit_1 + (0.8)*profit_2
    item_str = "\n item "+str(index+1)
    item_str = item_str+"\n  weight: +"+str(1)
    item_str = item_str+"\n  profit: +"+str(int(profit+0.5))
    strheader = strheader+item_str

print(strheader)

f = open('knapsack.100.3', 'w')
f.write(strheader)
f.close()

