import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Model.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
import DataStorage.FileUploader as fp





# define rate limit handler function
if __name__ == '__main__':
   
    n = 300
    P = 0.3
    K = 100
    M = 20
    nbb = 0
    NbrSim = 1

    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.7,
                  "jug_max": 0.99,
                  "beta_max": 0.6,
                  "beta_min": 0.1}
  

    # Generator=gg.CreateGraphFrmDB(uri="bolt://localhost:7687",username="neo4j",password="Graid4154")
    # g = Generator.CreateGraph(parameters,graphModel='ABS') 

    Generator=gg.CreateGraphFrmDB(uri="bolt://localhost:7687",username="neo4j",password="Graid4154")
    g = Generator.CreateGraph(parameters,graphModel='AB')  
    
    Simulator = sim.RumorSimulator()
   
   
    print("--------------------------------------------------------------------------------------------------------------------")
    start_time = time.time()
     
    typeOfSim=1
    k=int(0.05*g.number_of_nodes())
    i=0
    blockPeriod=2
    # aux1 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_RBN',k=k)
    
    aux2 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_DMBN',blockPeriod=blockPeriod,k=k)
    
    aux3 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='T_MDTCS',k=k)
    
    #aux4 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.03, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='T_MRIBHBTCS',k=k)
    
    #aux_0 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.02, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='None',k=k)
      

    l=[aux2,aux3]
    #l=aux4

    end_time = time.time()
    print('Parallel time: ', end_time-start_time)
   
    
    Simulator.DisplyResults( l,resultType=typeOfSim,save=False,imageName="")
  
    


    print("End Main Program")
    

    