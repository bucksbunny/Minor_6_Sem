
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sets import Set
import random as rd


# In[132]:

class_CBO = np.genfromtxt('class_weight.txt', dtype=None)
#class_CBO = list(class_CBO)

class_History = np.genfromtxt('history_list.txt', dtype=None)
#class_History = list(class_History)

class_combined_factors = []
for i in np.arange(0, class_CBO.shape[0]):
    name = class_CBO[i][1]
    CBO = float(class_CBO[i][2])
    History = float(class_History[i][1])
    total_factor = 0.2*CBO + 0.8*History
    combined_factors = [name, CBO, History, total_factor]
    class_combined_factors.append(combined_factors)
    
dataframe = pd.DataFrame(class_combined_factors, columns=['ClassName', 'ClassCBO', 'ClassHistory', 'TotalScore'])

dataframe.to_csv('TotalScore.csv', index=False, header=False)

print("For maximum TotalScore : ")
print(dataframe.loc[dataframe['TotalScore'] == max(dataframe.TotalScore)])

print("\n\nFor maximum CBO : ")
print(dataframe.loc[dataframe['ClassCBO'] == max(dataframe.ClassCBO)])

print("\n\nFor maximum History : ")
print(dataframe.loc[dataframe['ClassHistory'] == max(dataframe.ClassHistory)])

