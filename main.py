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
    type=1

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
    
    if(type==1):
      g = gg.CreateGraph(parameters, 1000,10)
      results = m.Simulations(5, g,seedsSize=0.05, typeOfSim= type)
      m.Create_Data_Globale(results)
    else:
        for beta in np.arange(0.1,1,0.05):
            parameters['beta_min']=beta
            parameters['beta_max']=beta+0.1
            g = gg.CreateGraph(parameters, 1000,10)
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
    
