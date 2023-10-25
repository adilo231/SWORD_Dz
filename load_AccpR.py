import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Simulation.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
import DataStorage.loadData2DB as fp
from Models import Models
import torch
from torch_geometric.utils import from_networkx

# define rate limit handler function
if __name__ == '__main__':
    parameters = {'omega_min': np.pi/24,
                    'omega_max': np.pi*2,
                    "delta_min": np.pi/24,
                    "delta_max": np.pi/2,
                    "jug_min": 0.01,
                    "jug_max": 0.99,
                    "beta_max": 1.2,
                    "beta_min": 0.2}

    Generator=gg.CreateGraphFrmDB()
    g = Generator.CreateGraph(parameters,graphModel='FB')
    """ for node, attrs in g.nodes(data=True):
        print(node, attrs) """
    """ Simulator = sim.RumorSimulator()
    # Run the simulation
    typeOfSim=0
    k=int(0.15*g.number_of_nodes())
    i=0
    blockPeriod=10
    NbrSim = 20
    df = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.03, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='none',k=k,blockPeriod=blockPeriod)
    for index, row in df.iterrows():    
        acceptance = row['AccpR']
        if index in g:
            g.nodes[index]['AccpR'] = acceptance
            #print(g.nodes[index]['AccpR'])
            
    acceptance = [g.nodes[node].get('AccpR', 0) for node in g.nodes()]
    acceptance_tensor = torch.tensor(acceptance, dtype=torch.float) """

    # Initialize the model first
    model = Models.GCN(num_features=5, hidden_channels=16, num_classes=1)

    # Load parameters
    model.load_state_dict(torch.load('GCN_87.81.pth'))
    # Initialize an empty list to hold feature vectors
    features_list = []

    # Loop through each node in the NetworkX graph
    for node in g.nodes():
        # Extract or compute the features for the node
        # For demonstration, let's say the feature vector for each node is [attribute1, attribute2]
        attribute1 = g.nodes[node].get('degree', 0)  # Replace 'attribute1' with your actual attribute name
        attribute2 = g.nodes[node].get('degree_centrality', 0)  # Replace 'attribute2' with your actual attribute name
        attribute3 = g.nodes[node].get('closeness_centrality', 0)
        attribute4 = g.nodes[node].get('between_centrality', 0)
        attribute5 = g.nodes[node].get('page_rank', 0)
        feature_vector = [attribute1, attribute2,attribute3,attribute4,attribute5]
        
        # Append the feature vector to the list
        features_list.append(feature_vector)

    # Convert the list of feature vectors to a PyTorch tensor
    features_tensor = torch.tensor(features_list, dtype=torch.float)

    data = from_networkx(g)
    data.x = features_tensor

    model.eval()
    with torch.no_grad():
        out = model(data)
    AccpR_List = [value.item() for value in out]
    print(AccpR_List)
    print('started')
    Generator1=fp.FileUploader()
    Generator1.add_AccpR(g,graphModel="FB",AccpR_Tensor=AccpR_List)
    print("fb is finiched")