import numpy as np
import networkx as nx

def GetRandomValues(n, min, max):
        return (np.random.rand(n)*(max - min)) + min


class CreateSytheticGraph():
        
    def CreateGraph(self,parameters,graphModel='AB', Graph_size=100, M=3):
        ''' Create a sythetic graph'''
        if graphModel== 'AB' or graphModel== 'barabasi_albert':
            g = nx.barabasi_albert_graph(Graph_size, M)
        self.InitParameters(g, parameters)
        return g

    # Init the model paramters
    def InitParameters(self,g, parameters):
        n = g.number_of_nodes()

        # Set omega

        values = dict(enumerate(GetRandomValues(
            n, parameters['omega_min'], parameters['omega_max'])))
        nx.set_node_attributes(g, values, 'omega')
        # Set beta
        values = dict(enumerate(GetRandomValues(
            n, parameters['beta_min'], parameters['beta_max'])))
        nx.set_node_attributes(g, values, 'beta')
        # Set delta
        values = dict(enumerate(GetRandomValues(
            n, parameters['delta_min'], parameters['delta_max'])))
        nx.set_node_attributes(g, values, 'delta')

        # Set jug
        values = dict(enumerate(GetRandomValues(
            n, parameters['jug_min'], parameters['jug_max'])))
        nx.set_node_attributes(g, values, 'jug')

        # Set other Attributes
        attributes = ["Infetime", "AccpR", "SendR", "Accp_NegR"]
        zeros = dict(enumerate(np.zeros(n)))
        for atrrib in attributes:
            nx.set_node_attributes(g, zeros, atrrib)

        nx.set_node_attributes(g, 'non_infected', "state")

        # S, D, Q, T: supporting, Denying, Questioning, Neutral
        nx.set_node_attributes(g, 'S', "opinion")

