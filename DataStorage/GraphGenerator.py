
import numpy as np
import networkx as nx
import pandas as pd
from pandas import DataFrame
import random 
from neo4j import GraphDatabase,basic_auth

def connexion():
    driver = GraphDatabase.driver(uri = "bolt://localhost:7687", auth=basic_auth("neo4j", "Graid4154"))
    session=driver.session()
    return session


def getRandomValues(n,min,max):
    return (np.random.rand(n)*(max - min)) + min

def InitParameters(g,parameters):
    n=g.number_of_nodes()

    # Set omega
    
    values = dict( enumerate(getRandomValues(n,parameters['omega_min'],parameters['omega_max'])) )
    nx.set_node_attributes(g, values, 'omega')
    # Set beta
    values = dict( enumerate(getRandomValues(n,parameters['beta_min'],parameters['beta_max'])) )
    nx.set_node_attributes(g, values, 'beta')
    # Set delta
    values = dict( enumerate(getRandomValues(n,parameters['delta_min'],parameters['delta_max'])) )
    nx.set_node_attributes(g, values, 'delta')

    # Set jug
    values = dict( enumerate(getRandomValues(n,parameters['jug_min'],parameters['jug_max'])) )
    nx.set_node_attributes(g, values, 'jug')

    # Set other Attributes
    attributes =[ "Infetime","AccpR","SendR","Accp_NegR"]
    zeros = dict( enumerate(np.zeros(n)) )
    for atrrib in attributes:
        nx.set_node_attributes(g, zeros, atrrib)

    nx.set_node_attributes(g, 'non_infected', "state")
 
    # S, D, Q, T: supporting, Denying, Questioning, Neutral
    nx.set_node_attributes(g, 'S', "opinion")


# create the facebook graph
def load_facebook_data_from_file(filename="./facebook.txt"):
    l=[]
    try:
        with open(filename) as file:
            for line in file:
                words=[]
                for word in line.split():
                    words.append(word)
                l.append(words)
            return l
    except:
        return l

def load_facebook_data_from_neo4j():
    l=[]
    try:
        session=connexion()
        ql="MATCH (u1:user)-[r:friend]->(u2:user) return distinct u1.id_user,u2.id_user"
        dtf_data = DataFrame([dict(_) for _ in session.run(ql)])
        l=dtf_data.values.tolist()
        return l
    except:
        return l

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



