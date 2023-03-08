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
    NbrSim =10

    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.7,
                  "jug_max": 0.99,
                  "beta_max": 0.6,
                  "beta_min": 0.1}
  

    Generator=gg.CreateGraphFrmDB()
    Simulator = sim.RumorSimulator()
    g = Generator.CreateGraph(parameters,graphModel='FB')  
  
   
    print("--------------------------------------------------------------------------------------------------------------------")
    start_time = time.time()
    
    typeOfSim=1
    k=int(0.0001*g.number_of_nodes())
    i=0
    setptime=0.125
    #aux1 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.006, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='B_BMDB',k=k)
    
    #aux2 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.006, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='T_MDBTCS',k=k)
    
    # aux3 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='T_MDTCS',k=k)
    
    # aux4 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='T_MRIBHBTCS',k=k)
    
    aux_0 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.002, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='None',k=k,setptime=setptime)
      
    l=[aux_0]#,aux1,aux2]


    end_time = time.time()
    print('Parallel time: ', end_time-start_time)
   
    
    Simulator.DisplyResults( l,resultType=typeOfSim,save=False)
  
    


    print("End Main Program")
    

    