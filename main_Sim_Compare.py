import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Simulation.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
import torch




# define rate limit handler function
if __name__ == '__main__':
   
    n = 300
    P = 0.3
    K = 100
    M = 20
    nbb = 0
    NbrSim = 5

    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.1,
                  "jug_max": 0.9,
                  "beta_max": 1.2,
                  "beta_min": 0.1}
  

   

    Generator=gg.CreateGraphFrmDB()
    g = Generator.CreateGraph(parameters,graphModel='FB')  
    
    Simulator = sim.RumorSimulator()
    # Run the simulation
    
    print("--------------------------------------------------------------------------------------------------------------------")
    start_time = time.time()
     
    typeOfSim=2
    k=int(0.15*g.number_of_nodes())
    i=1
    blockPeriod=10
    # aux1 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_RBN',k=k)
    
    # aux2 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_DMBN',blockPeriod=blockPeriod,k=k)
    
    #aux3 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_RBN',k=k,blockPeriod=blockPeriod,Tdet=1)
    
    aux4 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_ARM',k=k,blockPeriod=blockPeriod,Tdet=1)
    
    aux_0 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_BCN',k=k,blockPeriod=blockPeriod,Tdet=1)

    aux_2 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_RBN',k=k,blockPeriod=blockPeriod,Tdet=1)

    l=[aux4,aux_0,aux_2]#,aux1,aux2,aux3,aux4]
  

    end_time = time.time()
    print('Parallel time: ', end_time-start_time)
   
    
    Simulator.DisplyResults( l,resultType=typeOfSim,save=False,imageName="")
    """ # Iterate through the DataFrame and print the 'Age' attribute
    for index, row in df.iterrows():
        AccpR = row['AccpR']
        print(f"{index} has acceptance of {AccpR}.")
 """
    