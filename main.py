
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

            
                query=' #عيد_الفطر_المبارك'
                Query={}
                Query['query'] = f"  ({query})"
                

                # Query['query'] = """( الجزائر OR Algérie OR algerie OR #Algérie OR ALGERIA OR Algeria) AND (fiat OR Fiat OR FIAT OR voiture OR #Fiat  OR فيات)"""
                
                # Query['query'] = """كمال رزيق"""
                Query['query'] = """  #عيد_الفطر_المبارك"""
               

                Query['lang']='*'
                mongo_db = "twitter_db"
                mongo_tweet_collection = 'Aiide'
                mongo_user = f"AlgeriaTwitterGraph"
                Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)

            # query = "MATCH (u:User {Checked: False})   RETURN u.id_str AS id"
            
            # # Retrieve user IDs from Neo4j that hasn't been checked
            # Extractor.Graph_Extraction(mongo_db,mongo_user,query,verbose=True)


        elif num==1:
            Extractor =TweetExtractor(0)
            if num2:
                
                Query={}
               


                start_date = '2023-01-01'
                end_date = '2023-03-31'
                keywords = ["الجزائر" , "Algérie" , "algerie" , "#Algérie" , "ALGERIA" , "Algeria", "#Algeria"]
                keyword_query = ' OR '.join(keywords)
                keyword_query='(fiat OR Fiat OR FIAT)'+' AND '+keyword_query

                full_query = f"({keyword_query})  since:{start_date} until:{end_date}"

                Query['query'] = f" since:{start_date} until:{end_date} ({keyword_query})"


                 
                

                Query['lang']='*'
                mongo_db = "twitter_db"
                mongo_tweet_collection = 'FIAT2'
                mongo_user = f"AlgeriaTwitterGraph"
                Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)
            
            # query="MATCH (u:User) WHERE u.cursor_followers <> 0 AND u.cursor_followers <> -1 RETURN u.id_str as id "
            # # Retrieve user IDs from Neo4j that hasn't been checked
            # Extractor.Graph_Extraction(mongo_db,mongo_user,query,verbose=True)

        elif num==2:
            Extractor =TweetExtractor(2)
            if num2:
                Query={}
                # (change AND algérie)
                # Query['query'] = """#أحمد_عطاف"""
                Query['query'] = """Attaf"""
                Query['lang']='*'
                mongo_db = "twitter_db"
                mongo_tweet_collection = 'Attaf'
                mongo_user = f"AlgeriaTwitterGraph"
                Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)

            # query="MATCH (p:User{Checked: false})-[r:FOLLOWS]->({id_str:$id})RETURN p.id_str as id "
            # # Retrieve user IDs from Neo4j that hasn't been checked
            # Extractor.Graph_Extraction(mongo_db,mongo_user,query,verbose=True)

    except ValueError:
            print("Invalid integer provided.")
            sys.exit(1)


        


    



    

# if __name__ == '__main__':   
#     Extractor =TweetExtractor(0)
#     mongo_db = "twitter_db"
#     collection = "OnStreamTweets"
#     Extractor.Get_Tweets_in_streaming(mongo_db,collection)
    