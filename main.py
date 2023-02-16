import Model.HISBmodel as m
import DataStorage.GraphGenerator as gg
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
import time
import DataStorage.GraphGenerator as gg

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
    # print(seedNode,seedOpinion,len(seedNode),len(seedOpinion))
    # print(g.nodes[0]['jug'])
    # sim=HSIBmodel(g,seedNode,seedOpinion)
    # sim.runModel()
    # sim.DisplyResults()

    # # Run multiple and paralle simulations than display
    # start_time = time.time()
    # dfs = Simulations(3, g, seedNode, seedOpinion, 1)
    # end_time = time.time()
    # print('Parallel time: ', end_time-start_time)
    # DisplyResults(dfs)

    #Run multiple and paralle simulations get final results
    start_time = time.time()
    parameters = {'omega_min': np.pi/24,
                  'omega_max': np.pi*2,
                  "delta_min": np.pi/24,
                  "delta_max": np.pi/2,
                  "jug_min": 0.7,
                  "jug_max": 0.99,
                  "beta_max": 1.2,
                  "beta_min": 0.8}
    SimulationResults= pd.DataFrame()
    for beta in np.arange(0.1,1,0.05):
        parameters['beta_min']=beta
        parameters['beta_max']=beta+0.1
        g = m.CreateGraph(parameters, n)
        results = m.Simulations(5, g,seedsSize=0.05, typeOfSim= 2)
        SimulationResults = m.CreateDataFrame(results,SimulationResults,sim=f'beta=[{beta},{beta+0.1}]')
        
    end_time = time.time()
    print('Parallel time: ', end_time-start_time)
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name, ax in zip(['Infected','Suporting','Denying'], axes):
        sns.boxplot(data=SimulationResults, x='sim', y=name, ax=ax)
        ax.set_ylabel('Number of individuals')
        ax.set_title(name)
    # Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')
    plt.show()
   

    # Get the DataFrame results from simulation
    # for x in pipe_list:

    #  print((x.recv().shape))
    gg.printGraph()