import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sets import Set
import random as rd


def generateGraph(graph_pts, graph_dict):
    ''' Function to construct the graph using graph points and dictionary.
    We make use of NetworkX Graph library for python 2.
    '''

    G = nx.Graph()

    for a, b, wt in graph_pts:
        G.add_edge(graph_dict[a], graph_dict[b], weight=int(wt))
        
    return G


graph_pts = np.genfromtxt('output.txt', dtype=None)


### Now, create a list of all edges that will be in the graph
new_graph_edges = []

for i in np.arange(0, graph_pts.shape[0]):
    
    temp_str = graph_pts[i][1]
    s1, s2, s3 = temp_str.rpartition('.')
    
    # Any '*' import is not considered since it is a generic import
    if s3 == "*":
    	continue

    j1, j2, j3 = s1.partition('.')
    t1, t2, t3 = graph_pts[i][0].rpartition('.')
    
    # Any imports of kind 'java.something something' are not considered
    if j1 != "java" :
        new_graph_edges.append([t1, s3, int(1)])



final_graph_edges = np.asarray(new_graph_edges)
    
np.savetxt('new_graph_edges.txt', final_graph_edges, delimiter='\t', fmt="%s")


### Make a dictionary for all classes, so that it is easier to access

set_temp = Set()

for i,j,wt in final_graph_edges:
    set_temp.add(i)
    set_temp.add(j)

print("Number of graph edges : {}".format(final_graph_edges.shape[0]))
print("Number of distinct classes : {}".format(len(set_temp)))

graph_pts_dict = {}
reverse_graph_pts_dict = {}
temp_class_list = []
for num, class_name in enumerate(set_temp):
    graph_pts_dict[class_name] = num
    reverse_graph_pts_dict[num] = class_name
    temp_class_list.append([num, class_name])

class_list = np.asarray(temp_class_list)
np.savetxt('class_list.txt', class_list, delimiter='\t', fmt="%s")


### Now, construct the actual graph

G = generateGraph(final_graph_edges, graph_pts_dict)

print("Number of graph nodes : {}".format(len(G.nodes())))


### Find coupling metric value (CBO - Coupling Between Objects)for each class
node_weight = []

for node in G.nodes():
	node_weight.append([node, reverse_graph_pts_dict[node], len(G[node])])

node_weight_dtype = [('node1', int), ('node2', str), ('edge_count', int)]
node_weight = sorted(node_weight, key=lambda a_entry: a_entry[2])
node_weight_np = np.asarray(node_weight)
np.savetxt('class_weight.txt', node_weight, delimiter='\t', fmt="%s")