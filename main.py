
from DataExtraction.TwitterExtractor import  TweetExtractor
import sys

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

    if len(sys.argv) < 2:
        print("Usage: python script.py <integer>")
        sys.exit(1)

    # Convert the argument to an integer
    try:
        num = int(sys.argv[1])
        num2 = int(sys.argv[2])
        Locations=[ 'Algérie','Algiers','Alger','Algeria','الجزائر','Alger-Algérie','Algiers, Algeria']
        mongo_db = "twitter_db"
        mongo_user = "AlgeriaTwitterGraph"
    

        if num==0:
            Extractor =TweetExtractor(0)
            if num2:
                
                Query={}
                # Query['query'] = """( الجزائر OR Algérie OR algerie OR #Algérie OR ALGERIA OR Algeria) AND (fiat OR Fiat OR FIAT OR voiture OR #Fiat  OR فيات)"""
                Query['query'] = """ (التمور AND الجزائرية ) OR
                                    (#المغاربة_يشوهون_التمور_الجزائرية) OR
                                    (#التمور_الجزائرية ) """

                Query['lang']='*'
                mongo_db = "twitter_db"
                mongo_tweet_collection = 'DATTE-DZ'
                mongo_user = f"AlgeriaTwitterGraph"
                Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)

            # query = "MATCH (u:User {Checked: False})   RETURN u.id_str AS id"
            
            # # Retrieve user IDs from Neo4j that hasn't been checked
            # Extractor.Graph_Extraction(mongo_db,mongo_user,query,verbose=True)


        elif num==1:
            Extractor =TweetExtractor(1)
            if num2:
                
                Query={}
                Query['query'] = """الجزائر OR Algérie OR algerie OR #Algérie OR ALGERIA OR Algeria #Algeria"""
                

                Query['lang']='*'
                mongo_db = "twitter_db"
                mongo_tweet_collection = 'Algeria'
                mongo_user = f"AlgeriaTwitterGraph"
                Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)
            
            query="MATCH (u:User) WHERE u.cursor_followers <> 0 AND u.cursor_followers <> -1 RETURN u.id_str as id "
            # Retrieve user IDs from Neo4j that hasn't been checked
            Extractor.Graph_Extraction(mongo_db,mongo_user,query,verbose=True)

        elif num==2:
            Extractor =TweetExtractor(2)
            if num2:
                Query={}
                Query['query'] = """Mahrez"""
                Query['lang']='*'
                mongo_db = "twitter_db"
                mongo_tweet_collection = 'Tebboune_me_present'
                mongo_user = f"AlgeriaTwitterGraph"
                Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)

            # query="MATCH (p:User{Checked: false})-[r:FOLLOWS]->({id_str:$id})RETURN p.id_str as id "
            # # Retrieve user IDs from Neo4j that hasn't been checked
            # Extractor.Graph_Extraction(mongo_db,mongo_user,query,verbose=True)

    except ValueError:
            print("Invalid integer provided.")
            sys.exit(1)


        


    




    #Graph's Parametres
    # n = 300
    # seedsSize=0.02
    # typeOfSim=2
    # NbrSim=5
    # P = 0.3
    # K = 0.1
    # M = 20
    # nbb = 0
    

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
    

    