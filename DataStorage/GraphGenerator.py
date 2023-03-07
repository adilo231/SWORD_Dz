import numpy as np
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase,basic_auth
# need three class 1 mother and two daughters , one for suthetic graphs the other one to extract from the DB

class Graph():

    def __init__(self):
        self.graph = nx.Graph()
        self.clustring_coef=[]
        self.degree=[]
        self.degree_centrality=[]
        self.page_rank=[]
        self.between_centrality=[]
        self.closeness_centrality=[]
       

    def GetRandomValues(self,n, min, max):
        return (np.random.rand(n)*(max - min)) + min

     # Init the model paramters
   
    def InitParameters(self,g, parameters):
        n = g.number_of_nodes()

        # Set omega

        values = dict(enumerate(self.GetRandomValues(
            n, parameters['omega_min'], parameters['omega_max'])))
        nx.set_node_attributes(g, values, 'omega')
        # Set beta
        values = dict(enumerate(self.GetRandomValues(
            n, parameters['beta_min'], parameters['beta_max'])))
        nx.set_node_attributes(g, values, 'beta')
        # Set delta
        values = dict(enumerate(self.GetRandomValues(
            n, parameters['delta_min'], parameters['delta_max'])))
        nx.set_node_attributes(g, values, 'delta')

        # Set jug
        values = dict(enumerate(self.GetRandomValues(
            n, parameters['jug_min'], parameters['jug_max'])))
        nx.set_node_attributes(g, values, 'jug')

        # Set other Attributes
                     
        attributes = ['Infetime','AccpR','SendR','Accp_NegR','Nb_Accpted_Rm']
        zeros = dict(enumerate(np.zeros(n)))
        for atrrib in attributes:
            nx.set_node_attributes(g, zeros, atrrib)

        nx.set_node_attributes(g, 'non_infected', "state")
        nx.set_node_attributes(g, 'false', "blocked")

        # S, D, Q, T: supporting, Denying, Questioning, Neutral
        nx.set_node_attributes(g, 'S', "opinion")

        init=0
        nx.set_node_attributes(g, init, 'clustring_coef')
        for i in range(g.number_of_nodes()):
            g.nodes[i]['clustring_coef']=self.clustring_coef[i]
 
        nx.set_node_attributes(g, init, 'degree')
        for i in range(g.number_of_nodes()):
            g.nodes[i]['degree']=self.degree[i]

        nx.set_node_attributes(g, init, 'degree_centrality')
        for i in range(g.number_of_nodes()):
            g.nodes[i]['degree_centrality']=self.degree_centrality[i]

        nx.set_node_attributes(g, init, 'page_rank')
        for i in range(g.number_of_nodes()):
            g.nodes[i]['page_rank']=self.page_rank[i]

        nx.set_node_attributes(g, init, 'between_centrality')
        for i in range(g.number_of_nodes()):
            g.nodes[i]['between_centrality']=self.between_centrality[i]

        nx.set_node_attributes(g, init, 'closeness_centrality')
        for i in range(g.number_of_nodes()):
            g.nodes[i]['closeness_centrality']=self.closeness_centrality[i]

        
    def CreateGraph(self,parameters,graphModel):
        pass



class CreateSytheticGraph(Graph):
        
    def CreateGraph(self,parameters,graphModel='AB', Graph_size=100, M=3):
        ''' Create a sythetic graph'''
        if graphModel== 'AB' or graphModel== 'barabasi_albert':
            g = nx.barabasi_albert_graph(Graph_size, M)
            
            self.degree=nx.degree(g)
            self.degree_centrality=nx.degree_centrality(g)
            self.between_centrality=nx.betweenness_centrality(g)
            self.clustring_coef=nx.clustering(g)
            self.page_rank=nx.pagerank(g,alpha=0.8)
            self.closeness_centrality=nx.closeness_centrality(g)
           
        self.InitParameters(g, parameters)
        return g



class CreateGraphFrmDB(Graph):
    def __init__(self,uri="bolt://localhost:7687",username="neo4j",password="admin"):
        self.uri=uri
        self.username=username
        self.password=password

    def CreateGraph(self,parameters,graphModel):
        ''' load the facebook graph''' 
        g =self.loadGraph(graphModel)
        self.InitParameters(g, parameters)
        return g

    def getConnection(self):
        driver = GraphDatabase.driver(uri =self.uri, auth=basic_auth(self.username, self.password))
        session=driver.session()
        print("Seccessfully connected to Database: "+self.uri)
        return session

    def loadGraph(self,graphModel):

        session=self.getConnection()
        query=""
        if graphModel== 'FB' :
            query ="MATCH (u1:user)-[r:friend]->(u2:user) return distinct u1.id_user,u2.id_user"

        if graphModel== 'ABS' :
            query ="MATCH (u1:user_small_random)-[r:friend_in_ABS]->(u2:user_small_random) return distinct u1.id_user,u2.id_user"
        if graphModel== 'ABM' :
            query ="MATCH (u1:user_medium_random)-[r:friend_in_ABM]->(u2:user_medium_random) return distinct u1.id_user,u2.id_user"
        if graphModel== 'ABL' :
            query ="MATCH (u1:user_large_random)-[r:friend_in_ABL]->(u2:user_large_random) return distinct u1.id_user,u2.id_user"
        
        extrat_query1 =",u1.degree,u1.degree_centrality,u1.closness_centrality,u1.between_centrality,u1.page_rank,u1.clustering"
        extrat_query2 =",u2.degree,u2.degree_centrality,u2.closness_centrality,u2.between_centrality,u2.page_rank,u2.clustering"
        if query !="":
            query=query+extrat_query1+extrat_query2
            print(query)
            dtf_data = pd.DataFrame([dict(_) for _ in session.run(query)])
            l=dtf_data.values.tolist()
        else:
            return None
        nodes=[]
        g=nx.Graph()
        for line in l:
            u1=int(line[0])
            u2=int(line[1])
            g.add_nodes_from([u1,u2])
            g.add_edge(u1,u2)
            
            node1=[int(line[0]),int(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7])]
            node2=[int(line[1]),int(line[8]),float(line[9]),float(line[10]),float(line[11]),float(line[12]),float(line[13])]
            nodes.append(node1)
            nodes.append(node2)
        
        nb_nodes=g.number_of_nodes()
        self.degree=np.zeros(nb_nodes)
        self.degree_centrality=np.zeros(nb_nodes)
        self.closeness_centrality=np.zeros(nb_nodes)
        self.between_centrality=np.zeros(nb_nodes)
        self.page_rank=np.zeros(nb_nodes)
        self.clustring_coef=np.zeros(nb_nodes)

        for node in nodes:
            i=node[0]
            self.degree[i]=node[1]
            self.degree_centrality[i]=node[2]
            self.closeness_centrality[i]=node[3]
            self.between_centrality[i]=node[4]
            self.page_rank[i]=node[5]
            self.clustring_coef[i]=node[6]  
        self.graph=g
       
    
        return g 
    
    


if __name__ == '__main__':

    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.7,
                  "jug_max": 0.99,
                  "beta_max": 0.6,
                  "beta_min": 0.1}

    gg=CreateSytheticGraph()
    g=gg.CreateGraph(parameters,'AB')
    print ("nb nodes: ",g.number_of_nodes())