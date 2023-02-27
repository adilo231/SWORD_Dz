import DataStorage.GraphGenerator as gg
import Model.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm


if __name__ == '__main__':

    # Graph's Parametres
    n = 300
    P = 0.3
    K = 100
    M = 20
    nbb = 0
    NbrSim = 50

    # parameters = {'omega_min': np.pi/24,
    #               'omega_max': np.pi*2,
    #               "delta_min": np.pi/24,
    #               "delta_max": np.pi/2,
    #               "jug_min": 0.1,
    #               "jug_max": 0.4,
    #               "beta_max": 1.2,
    #               "beta_min": 0.05}
    # print('graphe generation')
    # g = CreateGraph(parameters, n)
    # seed = int(0.05*n)
    # l = ['D', 'S']
    # seedNode = random.sample(range(0, n), seed)
    # seedOpinion = random.choices(l, k=seed)
    # print('simulation')

    # run simple simulation and display

    # sim=HSIBmodel(g,seedNode,seedOpinion)
    # sim.runModel()
    # sim.DisplyResults()
    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.7,
                  "jug_max": 0.99,
                  "beta_max": 0.6,
                  "beta_min": 0.1}
    # # Run multiple and paralle simulations than display
    # Generator=gg.CreateGraphFrmDB()
    # Simulator = sim.RumorSimulator()
    # Attr_list=[]

    # g = Generator.loadFecebookGraph()     
    Generator=gg.CreateSytheticGraph()
    Simulator = sim.RumorSimulator()
    Attr_list=[]

    g = Generator.CreateGraph(parameters,graphModel='AB',Graph_size=n)  
    start_time = time.time()
    df=pd.DataFrame()
    
    i=0
    l=[]
    l2=[]
    Generator.InitParameters(g,parameters)
    aux1 = Simulator.runSimulation(g, NbrSim=5 ,seedsSize=0.05, typeOfSim=1,simName=f'sim{i}',verbose=True,method='RBN',k=50)
    aux2 = Simulator.runSimulation(g, NbrSim=5 ,seedsSize=0.05, typeOfSim=1,simName=f'sim{i}',verbose=True,method='BBN',k=50)
    aux3 = Simulator.runSimulation(g, NbrSim=5 ,seedsSize=0.05, typeOfSim=1,simName=f'sim{i}',verbose=True,method='DMBN',k=50)
    aux4 = Simulator.runSimulation(g, NbrSim=5 ,seedsSize=0.05, typeOfSim=1,simName=f'sim{i}',verbose=True,method='BCN',k=50)
    aux_0 = Simulator.runSimulation(g, NbrSim=5 ,seedsSize=0.05, typeOfSim=1,simName=f'sim{i}',verbose=True,method='non',k=50)
    print(aux1)   
    l=[aux_0,aux1,aux2,aux3,aux4]
    l2=["non","RBN","BBN","DMBN","BCN"]
    

    end_time = time.time()
    print('Parallel time: ', end_time-start_time)

    Simulator.DisplyResults( l,l2,resultType=1)
  
    print(df)


    print("End Main Program")
    

    