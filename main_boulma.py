import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Simulation.Simulator as sim
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
    NbrSim = 1

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
    Simulator = sim.RumorSimulator()
   
   
    print("--------------------------------------------------------------------------------------------------------------------")
    start_time = time.time()
    typeOfSim=0
    k=int(0.1*g.number_of_nodes())


    g = Generator.CreateGraph(parameters,graphModel='FB')
    aux = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.005, typeOfSim=typeOfSim,simName=f'sim1',verbose=False,method='None',k=int(0.1*g.number_of_nodes()),Tdet=1000000)

    end_time = time.time()
    print('Parallel time: ', end_time-start_time)

   
    print(aux.shape)
    Simulator.DisplyResults( aux,resultType=typeOfSim,save=False,imageName="")
  
    


    print("End Main Program")
    

    