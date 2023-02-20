

def printGraph():
    print('hello')

def facebook_graph():
    g=nx.Graph()
    l=load_facebook_data_from_neo4j()
    for line in l:
        a=int(line[0])
        b=int(line[1])
        g.add_nodes_from([a,b])
        g.add_edge(a,b)
            
    return g 

def CreateGraph(parameters,N=100,M=3):
    #g=nx.barabasi_albert_graph(N,M)
    g=facebook_graph()
    InitParameters(g,parameters)
    return g



