import networkx as nx
from networkx.readwrite import json_graph
from scipy.io import loadmat
import os
import random
from networkx import average_clustering,average_shortest_path_length

#this function returns a directed graph
def txt2Graph(FielName):
	# H = nx.path_graph(6408)
	# g=nx.DiGraph()
	# g.add_nodes_from(H)
	# g.remove_node(0)
	Graphtype=nx.DiGraph()
	g0= nx.read_edgelist(FielName,create_using=Graphtype,nodetype=int)
	# g.add_edges_from(g0.edges())

	#print("hereeee:  ", len(list(g.successors(2787))))
	return graphe_TO_json_directedGraph(g0) 

def graphe_TO_json_directedGraph(g):
    
    data =  json_graph.node_link_data(g,{"link": "links", "source": "source", "target": "target","weight":"weight"})
    
    data['nodes'] = [ {"id": i['id'],"state":"non_infected","Protector":"false","opinion":"normal","beta":0,"omega":0,"delta":0,"jug":0,"Infetime":0,"AccpR":0,"SendR":0,"Accp_NegR":0,"value":0,"blocked":'false',"p0_drimux":random.random(),"BlockedTemporary":'false',"Blockingtime":0,"degree":g.degree[i['id']],"neighbors":[n for n in g.successors(i['id'])]} for i in data['nodes'] ]
    data['links'] = [ {"source":u,"target":v,"weight":(g.degree[u]+g.degree[v])/2} for u,v in g.edges ]
    return data


if __name__ == '__main__':
    # g1 = nx.DiGraph()
        
    g = json_graph.node_link_graph(txt2Graph('twitter.txt'))
    # print(len(g.edges)) 
    # for node in g.nodes:
    #     print("node: ", node['id'])
    '''
    with open('twitter.txt', 'w') as outfile:
        for fname in os.listdir('./twitter'):
            with open('./twitter/'+fname) as infile:
                for line in infile:
                    outfile.write(line)
    '''
   
    #print("average_shortest_path_length",average_shortest_path_length(g))

    Tdet=15

    
    clusteringC = round(average_clustering(g, weight=None), 4)
    print("clusteringC",clusteringC)
    print("k",len(g.edges)/len(g.nodes))
