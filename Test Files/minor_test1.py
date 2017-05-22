import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sets import Set
import random as rd

def node_valSum(G):
    
    sum_val = 0
    
    for node in G.nodes():
        temp_sum = 0
        for node1, node2 in G.edges(node):
            temp_sum += G[node1][node2]['weight']
        temp_sum /= len(G.edges(node))
        sum_val += temp_sum
    
    return sum_val
    
    
def HCS_Algo(G, subgraph_list, div_factor):
    ''' Implements a version of Highly Connected Subgraph (HCS) algorithm for weighted graphs
        
        Attribute 'div_factor' basically controls the minimum number of
        effective nodes that should be connected for a graph to be highly conneceted
        
        E.g. div_factor of 4 indicates that minimum cut should be more than n/4 for the graph to
        be highly connected. Here, n is the effective number of nodes (by taking into account the
        weights of edges that each node is connected to)
    '''
    
    if len(G.nodes()) <= 2:      ### Test variation here
        print('Did not split : {}'.format(len(G.nodes())))
        subgraph_list.append(G)
        return subgraph_list
    
    if nx.is_connected(G) == False:
        conn_components = nx.connected_components(G)
        conn_subgraphs = []
        for c in conn_components:
            conn_subgraphs.append(nx.subgraph(G, c))
        
        g_len_list = []
        
        ##### New for test

        '''for g in conn_subgraphs:
                                    print("Graph nodes: {}".format(g.nodes()))'''

        ##### Test End

        for g in conn_subgraphs:
            g_len_list.append(len(g.nodes()))
        
        print("Splitting graph into subgraphs of these size (nodes) : {}".format(g_len_list))
        
        for g in conn_subgraphs:
            subgraph_list = HCS_Algo(g, subgraph_list, div_factor)
        
        return subgraph_list

    print('Now Running Min Cut on graph of size (nodes) : {}'.format(len(G.nodes())))
    #cut_val, partition = nx.stoer_wagner(G)
    random_node_1 = rd.randrange(0, len(G.nodes()), 1)
    random_node_2 = rd.randrange(0, len(G.nodes()), 1)
    while random_node_2 == random_node_1:
        random_node_2 = rd.randrange(0, len(G.nodes()), 1)
    
    cut_val, partition = nx.minimum_cut(G, G.nodes()[random_node_1], G.nodes()[random_node_2]
                                       , capacity='weight')
    print('Min Cut complete')    

    G1 = nx.subgraph(G, partition[0]) # subgraph 1
    G2 = nx.subgraph(G, partition[1]) # subgraph 2

    #mincut = len(G.edges()) - (len(G1.edges()) + len(G2.edges()))  # Not useful if edge weights are considered
    ### Test variation here in the condition
    #if mincut >= len(G.nodes())/2:  # Check if it is highly connected
    
    '''print("\nNode sum = {:.2f}".format(node_valSum(G)))
    print(u"Node sum \u00F7 {} = {} and cut_val = {}".format(div_factor, 
                                                             node_valSum(G)/div_factor, cut_val))'''
    if cut_val >= len(G.nodes())/div_factor:
    #if cut_val >= node_valSum(G)/div_factor:
        #print('Didn\'t split : {}'.format(G.nodes()))
        print('Did not split : {}'.format(len(G.nodes())))
        subgraph_list.append(G)
        return subgraph_list
    else:
        #print('Now splittting : {}'.format(G.nodes()))

        print("Cut val : {}".format(cut_val))

        ##### New for test

        if len(G1.nodes()) < 5:
            print("Graph nodes: {}".format(G1.nodes()))
        else:
            print("Graph nodes: {}".format(G2.nodes()))

        

        ##### Test End


        print("Splitting graph into subgraphs of these size"+
              " (nodes) : {} {}".format(len(G1.nodes()), len(G2.nodes())))
        subgraph_list = HCS_Algo(G1, subgraph_list, div_factor)
        subgraph_list = HCS_Algo(G2, subgraph_list, div_factor)
        return subgraph_list
    

def generateGraph(graph_pts, graph_dict):
    G = nx.Graph()

    for a, b, wt in graph_pts:
        G.add_edge(graph_dict[a], graph_dict[b], weight=int(wt))
        
    return G



graph_pts = np.genfromtxt('output.txt', dtype=None)

new_graph_edges = []


count_java = 0;
for i in np.arange(0, graph_pts.shape[0]):
    
    temp_str = graph_pts[i][1]
    s1, s2, s3 = temp_str.rpartition('.')
    j1, j2, j3 = s1.partition('.')
    t1, t2, t3 = graph_pts[i][0].rpartition('.')
    if j1 != "java" :
        new_graph_edges.append([t1, s3, int(1)])
    else:
        count_java += 1
	

print("java Count :{}".format(count_java))
    
final_graph_edges = np.asarray(new_graph_edges)
    
np.savetxt('new_graph_edges.txt', final_graph_edges, delimiter='\t', fmt="%s")


# Make a dictionary for all classes. Since the stoer-wagner algo works well only with
# int values, hence we will use the integer key value for each class as per this dictionary

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


G = generateGraph(final_graph_edges, graph_pts_dict)

print("Number of graph nodes : {}".format(len(G.nodes())))

#print(G.edges())

#print(final_graph_edges)


subgraph_list = []

node_weight = []

for node in G.nodes():
	node_weight.append([node, reverse_graph_pts_dict[node], len(G[node])])
	#print("Node edge size : {}".format(len(G[node])))

node_weight_dtype = [('node1', int), ('node2', str), ('edge_count', int)]
#node_weight_np = np.asarray(node_weight)
node_weight = sorted(node_weight, key=lambda a_entry: a_entry[2])
node_weight_np = np.asarray(node_weight)
#node_weight_np.view('i8,i8,i8').sort(order=['f2'], axis=0)
np.savetxt('class_weight', node_weight, delimiter='\t', fmt="%s")


'''subgraph_list = HCS_Algo(G, subgraph_list, 4)


subgraph_list_sizes = []

for g in subgraph_list:
    subgraph_list_sizes.append(len(g.nodes()))

print('Subgraph node sizes : {}'.format(subgraph_list_sizes))
'''

'''#### TEST new

lololol = []
lololol.append([G.nodes()[0], reverse_graph_pts_dict[G.nodes()[0]]])
lalalal = np.asarray(lololol)
print(lalalal)
rar = np.arange(0,3)
print(rar)
for g, i in zip(G.nodes(), rar):
    np.savetxt('sub_graph_list_'+str(i)+'.txt', lalalal, delimiter='\t', fmt="%s")

#### End Test new'''

'''for sg, i in zip(subgraph_list, len(subgraph_list)):
    if len(sg.nodes()) > 1:
        temp_graph_arr = []
        
        for node in sg.nodes():
            temp_graph_arr.append([node, reverse_graph_pts_dict[node]])

        graph_arr = np.asarray(temp_graph_arr)
        np.savetxt('sub_graph_list_'+str(i)+'.txt', graph_arr, delimiter='\t', fmt="%s")

'''