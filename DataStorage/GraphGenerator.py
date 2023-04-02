import numpy as np
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase, basic_auth
from DataStorage.DBHandlers import GraphDBHandler
import json
# need three class 1 mother and two daughters , one for suthetic graphs the other one to extract from the DB


class Graph():

    def __init__(self):
        self.graph = nx.Graph()
        self.clustring_coef = []
        self.degree = []
        self.degree_centrality = []
        self.page_rank = []
        self.between_centrality = []
        self.closeness_centrality = []

    def GetRandomValues(self, n, min, max):
        return (np.random.rand(n)*(max - min)) + min

     # Init the model paramters

    def InitParameters(self, g, parameters):
        """
        Initializes the parameters of a graph by setting various node attributes and metrics.

        Args:
            g (networkx.Graph): the graph to initialize
            parameters (dict): a dictionary of parameter values, including:
                - omega_min (float): the minimum value of omega
                - omega_max (float): the maximum value of omega
                - beta_min (float): the minimum value of beta
                - beta_max (float): the maximum value of beta
                - delta_min (float): the minimum value of delta
                - delta_max (float): the maximum value of delta
                - jug_min (float): the minimum value of jug
                - jug_max (float): the maximum value of jug

        Returns:
            None
        """
        n = g.number_of_nodes()

        # Set node attributes for omega, beta, delta, and jug
        for attribute in ['omega', 'beta', 'delta', 'jug']:
            values = dict(enumerate(self.GetRandomValues(
                n, parameters[f'{attribute}_min'], parameters[f'{attribute}_max'])))
            nx.set_node_attributes(g, values, attribute)

        # Set statistics attributes
        attributes = ['Infetime', 'AccpR',
                      'SendR', 'Accp_NegR', 'Nb_Accpted_Rm']
        zeros = dict(enumerate(np.zeros(n)))
        for atrrib in attributes:
            nx.set_node_attributes(g, zeros, atrrib)

        # Set Nodes proporeties  attributes
        nx.set_node_attributes(g, 'non_infected', "state")
        nx.set_node_attributes(g, 'false', "blocked")
        # S, D, Q, T: supporting, Denying, Questioning, Neutral
        nx.set_node_attributes(g, 'S', "opinion")
        nx.set_node_attributes(g, 0, 'blocking_time')

        # Add Nodes Metrics
        attribute = []
        for attribute in ['clustring_coef', 'degree', 'degree_centrality', 'page_rank', 'between_centrality', 'closeness_centrality']:
            nx.set_node_attributes(g, 0, attribute)
            for i in range(g.number_of_nodes()):
                g.nodes[i][attribute] = getattr(self, attribute)[i]

    def CreateGraph(self, parameters, graphModel):
        pass


class CreateSytheticGraph(Graph):

    def CreateGraph(self, parameters, graphModel='AB', Graph_size=100, M=3):
        ''' Create a sythetic graph'''
        if graphModel == 'AB' or graphModel == 'barabasi_albert':
            g = nx.barabasi_albert_graph(Graph_size, M)
        self.InitParameters(g, parameters)
        return g


class CreateGraphFrmDB(Graph):

    def __init__(self):
        Handler = GraphDBHandler()
        self.session = Handler.getConnection()

    def CreateGraph(self, parameters, graphModel):
        ''' load the facebook graph'''
        g = self.loadGraph(graphModel)
        self.InitParameters(g, parameters)

        return g

    def loadGraph(self, graphModel):

        query = ""
        if graphModel == 'FB':
            query = """MATCH (u1:user)-[r:friend]->(u2:user) return distinct u1.id_user,u2.id_user"""

        if graphModel == 'ABS':
            query = "MATCH (u1:user_small_random)-[r:friend_in_ABS]->(u2:user_small_random) return distinct u1.id_user,u2.id_user"
        if graphModel == 'ABM':
            query = "MATCH (u1:user_medium_random)-[r:friend_in_ABM]->(u2:user_medium_random) return distinct u1.id_user,u2.id_user"
        if graphModel == 'ABL':
            query = "MATCH (u1:user_large_random)-[r:friend_in_ABL]->(u2:user_large_random) return distinct u1.id_user,u2.id_user"

        extrat_query1 = ",u1.degree,u1.degree_centrality,u1.closness_centrality,u1.between_centrality,u1.page_rank,u1.clustering"
        extrat_query2 = ",u2.degree,u2.degree_centrality,u2.closness_centrality,u2.between_centrality,u2.page_rank,u2.clustering"
        if query != "":
            query = query+extrat_query1+extrat_query2
            dtf_data = pd.DataFrame([dict(_) for _ in self.session.run(query)])
            l = dtf_data.values.tolist()
        else:
            return None
        nodes = []
        g = nx.Graph()
        for line in l:
            u1 = int(line[0])
            u2 = int(line[1])
            g.add_nodes_from([u1, u2])
            g.add_edge(u1, u2)

            node1 = [int(line[0]), int(line[2]), float(line[3]), float(
                line[4]), float(line[5]), float(line[6]), float(line[7])]
            node2 = [int(line[1]), int(line[8]), float(line[9]), float(
                line[10]), float(line[11]), float(line[12]), float(line[13])]
            nodes.append(node1)
            nodes.append(node2)

        nb_nodes = g.number_of_nodes()
        self.degree = np.zeros(nb_nodes)
        self.degree_centrality = np.zeros(nb_nodes)
        self.closeness_centrality = np.zeros(nb_nodes)
        self.between_centrality = np.zeros(nb_nodes)
        self.page_rank = np.zeros(nb_nodes)
        self.clustring_coef = np.zeros(nb_nodes)

        for node in nodes:
            i = node[0]
            self.degree[i] = node[1]
            self.degree_centrality[i] = node[2]
            self.closeness_centrality[i] = node[3]
            self.between_centrality[i] = node[4]
            self.page_rank[i] = node[5]
            self.clustring_coef[i] = node[6]
        self.graph = g

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

    gg = CreateSytheticGraph()
    g = gg.CreateGraph(parameters, 'AB')
    print("nb nodes: ", g.number_of_nodes())
