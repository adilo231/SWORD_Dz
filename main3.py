from neo4j import GraphDatabase
from tqdm import  tqdm
from multiprocessing import Pool
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

auth = ("neo4j", "admin")

driver = GraphDatabase.driver(neo_uri, auth=auth, encrypted=False, max_connection_lifetime=3600)


# Define the function to execute the Neo4j query asynchronously
def execute_query(query, params):
     print(f"\t\t\t runing process Links  {params['follower_id']}")
     with driver.session() as session:
             session.run(query, params)
     print(f"\t\t\t runing process Links  {params['follower_id']} end")

# Define the main function to upload links to Neo4j
def upload_links(ids, user_id, mongo_user_collection):
    query = ("MERGE (a:User {id: $user_id}) "
             "MERGE (b:User {id: $follower_id, MongoCol: $mongo_col, Checked: $checked}) "
             "MERGE (b)-[:FOLLOWS]->(a)")
    print("\t Loading Links inside")
    follower_id_10=[]
    for follower_id in tqdm(ids):
        follower_id_10.append(str(follower_id))
        if len(follower_id_10)==10:
            print("\t\t run process Links inside",len(follower_id_10))

            

           
            
            pool = Pool(processes=len(follower_id_10))
            params_list = [{'user_id': str(user_id), 'follower_id': idf, 'mongo_col': mongo_user_collection, 'checked': False} for idf in follower_id_10]
            
            pool.map_async(execute_query, [(query, params) for params in params_list])
            pool.close()
            pool.join()
            
            
            follower_id_10=[]




# Call the main function with async/await








# define rate limit handler function
if __name__ == '__main__':

    Extractor =TweetExtractor(mongo_uri,neo_uri,neo_user,neo_password,API_credentials)
    mongo_user = "AlgeriaTwitterGraph"
    Extractor.UploadIdstoNeo4j(mongo_user)

 