import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Model.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm





# define rate limit handler function
if __name__ == '__main__':
   
    n = 300
    P = 0.3
    K = 100
    M = 20
    nbb = 0
    NbrSim = 2

    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.1,
                  "jug_max": 0.99,
                  "beta_max": 1.2,
                  "beta_min": 0.2}
  

   

    Generator=gg.CreateGraphFrmDB()
    g = Generator.CreateGraph(parameters,graphModel='FB')  
    
    Simulator = sim.RumorSimulator()
   
   
    print("--------------------------------------------------------------------------------------------------------------------")
    start_time = time.time()
     
    typeOfSim=1
    k=int(0.1*g.number_of_nodes())
    i=0
    blockPeriod=100

    l=[]
    methods=['B_BBN','B_DMBN','BCN','B_BeCN','MINJUGBN','B_BMDB','B_BMDBj','None']
    for method in tqdm( methods, position=0):

        aux = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.005, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=False,method=method,k=k)
        l.append(aux)

    # l=[]
  
    # # for t in tqdm( np.arange(0,10,2), position=0):

    # aux = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=False,method='None',k=int(0.1*g.number_of_nodes()),Tdet=1000000)
    # l.append(aux)



    end_time = time.time()
    print('Parallel time: ', end_time-start_time)
   
    
    Simulator.DisplyResults( l,resultType=typeOfSim,save=False,imageName="")
  
    


    print("End Main Program")
    

    