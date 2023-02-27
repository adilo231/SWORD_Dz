import numpy as np
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase,basic_auth
# need three class 1 mother and two daughters , one for suthetic graphs the other one to extract from the DB

class Graph():

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

    def CreateGraph(self,parameters,graphModel,Graph_size):
        pass



class CreateSytheticGraph(Graph):
        
    def CreateGraph(self,parameters,graphModel='AB', Graph_size=100, M=3):
        ''' Create a sythetic graph'''
        if graphModel== 'AB' or graphModel== 'barabasi_albert':
            g = nx.barabasi_albert_graph(Graph_size, M)
        self.InitParameters(g, parameters)
        return g



class CreateGraphFrmDB(Graph):

    def CreateGraph(self,parameters,graphModel, Graph_size=100, M=3):
        ''' Create a sythetic graph'''
        if graphModel== 'FB' :
            g =self.loadFecebookGraph()

        self.InitParameters(g, parameters)
        return g

    def getConnection(self,uri,username,password):
        driver = GraphDatabase.driver(uri =uri, auth=basic_auth(username, password))
        session=driver.session()
        print("Seccessfully connected to Database: "+uri)
        return session

    def loadFecebookGraph(self):
        uri="bolt://localhost:7687"
        username="neo4j"
        password="1151999aymana"
        session=self.getConnection(uri,username,password)

        query ="MATCH (u1:user)-[r:friend]->(u2:user) return distinct u1.id_user,u2.id_user"
        dtf_data = pd.DataFrame([dict(_) for _ in session.run(query)])
        l=dtf_data.values.tolist()

        g=nx.Graph()
        for line in l:
            a=int(line[0])
            b=int(line[1])
            g.add_nodes_from([a,b])
            g.add_edge(a,b)
            
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