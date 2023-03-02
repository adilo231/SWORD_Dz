import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Model.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm

# Twitter API credentials
API_credentials={
'consumer_key' : "2nyGsL1XyRXFGMw0MXbvm7cnY",
'consumer_secret' : "SPTXJkrZfukdZ4gnhYZVgOmpvEbE9Z1lKwxjRHOxbbucidENzs",
'access_key' : "2275955796-bcNPjJsfAWdzG1dJrd7AqjzoWgsEP83XgxYNIaq",
'access_secret' : "IhzSm6k7GyqndpqrSPgypQDIm93uMjaBRkRw431TABLSE"}



# MongoDB credentials
mongo_uri = "mongodb://localhost:27017/"

# Neo4j credentials
neo_uri = "bolt://localhost:7687"
neo_user = "neo4j"
neo_password = "admin"






# define rate limit handler function
if __name__ == '__main__':
    # Extractor =TweetExtractor(mongo_uri,neo_uri,neo_user,neo_password,API_credentials)
    # Query={
    #     'query' : "تبون",
    #     'lang': "en"

    # }
    

    # mongo_db = "twitter_db"
    # mongo_tweet_collection = "Teboune"
    # mongo_user_collection = "Teboune_users"
    # # Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user_collection)


    # Locations=[ 'Algérie','Algiers','Alger','Algeria']
    # mongo_db = "twitter_db"
    # mongo_user = "AlgeriaTwitterGraph"
    # Extractor.Graph_Extraction(mongo_db,mongo_user,Locations)
    # Graph's Parametres
    n = 300
    P = 0.3
    K = 100
    M = 20
    nbb = 0
    NbrSim = 2

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
    # # sim.DisplyResults()
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
    # Generator=gg.CreateSytheticGraph()
    # Simulator = sim.RumorSimulator()

    g = Generator.CreateGraph(parameters,graphModel='AB',Graph_size=n)  
    start_time = time.time()
    df=pd.DataFrame()
    
    i=0
    l=[]
    
    typeOfSim=1

    aux1 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='RTCS',k=30)
    # Generator.InitParameters(g,parameters)
    # aux2 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='BBN',k=50)
    # Generator.InitParameters(g,parameters)
    # aux3 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='DMBN',k=50)
    # Generator.InitParameters(g,parameters)
    # aux4 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='BCN',k=50)
    Generator.InitParameters(g,parameters)
    aux_0 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=0.05, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='non',k=30)
      
    l=[aux_0,aux1]#,aux2,aux3,aux4]

    #l=aux_0

    # end_time = time.time()
    # print('Parallel time: ', end_time-start_time)
    # if typeOfSim==0 or typeOfSim==2:
    #     l=aux_0
    
    Simulator.DisplyResults( l,resultType=typeOfSim)
  
    print(df)


    print("End Main Program")
    

    