
from DataExtraction.TwitterExtractor import  TweetExtractor


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

    # keywordslist=[]
    # keywordslist.append(["fertility", "vaccines", "covid"])
    # keywordslist.append(["CDC", "deaths", "covid"])
    

    # keywords=[ 'Algérie','Algiers','Alger','Algeria','الجزائر','جزائر','dz','DZ']
    # # Define the search query
    # query = " OR  ".join(keywords)
    
    # Query={
    #     'query' : query,
    # }



    # mongo_db = "twitter_db"
    # mongo_tweet_collection = "Algeria"
    # mongo_user_collection = f"Algeria_users"
    # Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user_collection,verbose=False)
    # Extractor =TweetExtractor(mongo_uri,neo_uri,neo_user,neo_password,API_credentials)
    # Query={
    #     'query' : "تبون",
    #     'lang': "en"

    # }
    

    # mongo_db = "twitter_db"
    # mongo_tweet_collection = "Teboune"
    # mongo_user_collection = "Teboune_users"
    # # Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user_collection)

# 1,237,920 2,912,785
# 1,264,045 3,055,290
# 1,301,232 3,274,314
# 1,625,307 4,443,375
# 1,756,211 4,987,064
    Locations=[ 'Algérie','Algiers','Alger','Algeria','الجزائر','Alger-Algérie','Algiers, Algeria']
    mongo_db = "twitter_db"
    mongo_user = "AlgeriaTwitterGraph"
    Extractor =TweetExtractor(mongo_uri,neo_uri,neo_user,neo_password,API_credentials)
    Extractor.Graph_Extraction(mongo_db,mongo_user,Locations)
    




    #Graph's Parametres
    n = 300
    seedsSize=0.02
    typeOfSim=2
    NbrSim=5
    P = 0.3
    K = 0.1
    M = 20
    nbb = 0
    

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
    # # sim.DisplyResult


    # parameters = {'omega_min': np.pi/24,
    #               'omega_max': np.pi*2,
    #               "delta_min": np.pi/24,
    #               "delta_max": np.pi/2,
    #               "jug_min": 0.7,
    #               "jug_max": 0.99,
    #               "beta_max": 0.6,
    #               "beta_min": 0.1}
  
    # Generator=gg.CreateSytheticGraph()
    # Simulator = sim.RumorSimulator()
    
 

    # g = Generator.CreateGraph(parameters,graphModel='AB',Graph_size=n)  
    # start_time = time.time()
    # df=pd.DataFrame()
    # print('')
    # i=0
    # l=[]
    # l2=[]
    
    # print(f'Number of simulation for each run {NbrSim}')
    # Generator.InitParameters(g,parameters)
    # aux1 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=seedsSize, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='RBN',k=int(n*K))
    # Generator.InitParameters(g,parameters)
    # aux2 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=seedsSize, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='BBN',k=int(n*K))
    # Generator.InitParameters(g,parameters)
    # aux3 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=seedsSize, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='DMBN',k=int(n*K))
    # Generator.InitParameters(g,parameters)
    # aux4 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=seedsSize, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='BCN',k=int(n*K))
    # Generator.InitParameters(g,parameters)
    # aux_0 = Simulator.runSimulation(g, NbrSim=NbrSim ,seedsSize=seedsSize, typeOfSim=typeOfSim,simName=f'sim{i}',verbose=True,method='non',k=int(n*K))
    # print(aux1)   
    # l=[aux_0,aux1,aux2,aux3,aux4]
    # l2=["none","RBN","BBN","DMBN","BCN"]
    

    # # end_time = time.time()
    # # print('Parallel time: ', end_time-start_time)

    # Simulator.DisplyResults( l,l2,resultType=1)
  
    # print(df)


    # print("End Main Program")
    

    