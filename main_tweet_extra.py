
from DataExtraction.TwitterExtractor import  TweetExtractor
# Twitter API credentials
API_credentials={
'consumer_key' : "dna9eBEhIe9oVIisEyUc8ZsHX",
'consumer_secret' : "0J7W6APSraCL4sJWvLSJyCGVOuHb5lPKyRK4NijZQQr0o0sSDm",
'access_key' : "2275955796-2kLj1PkrA1gfI3xffissxqYlTdHbibeafRsSMKb",
'access_secret' : "zbE7p91v1jGfXYXjA50PVwur9gGFVYNZkZ7cKKL7lAWvq"}



# MongoDB credentials
mongo_uri = "mongodb://localhost:27017/"

# Neo4j credentials
neo_uri = "bolt://localhost:7687"
neo_user = "neo4j"
neo_password = "admin"






# define rate limit handler function
if __name__ == '__main__':

    Extractor =TweetExtractor(2)  
    Query={}
    


    start_date = '2023-01-01'
    end_date = '2023-03-21'
    keywords = ["الجزائر" , "cycling" , "tour" , "#Algérie" , "ALGERIA" , "Algeria", "#Algeria","velo",'accident velo',"الدراجات"]
    keyword_query = ' OR '.join(keywords)
    Query['query'] = f" since:{start_date} until:{end_date} ({keyword_query})"
    #Query['query']="""Tour d'Algerie International de Cyclisme 2023"""

        
    

    #Query['lang']='*'
    mongo_db = "twitter_db"
    mongo_tweet_collection = 'تظاهرة الدراجات الجزائر'
    mongo_user = f"AlgeriaTwitterGraph"
    Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)

    
    # Extractor =TweetExtractor(2)
    # Query={}
    # # Query['query'] = """( الجزائر OR Algérie OR algerie OR #Algérie OR ALGERIA OR Algeria) AND (fiat OR Fiat OR FIAT OR voiture OR #Fiat  OR فيات)"""
    # # Query['query'] = """ (التمور AND الجزائرية ) OR
    # #                       (#المغاربة_يشوهون_التمور_الجزائرية) OR
    # #                       (#التمور_الجزائرية ) """

    # Query['query'] = """tour de velo algerien"""
                      
 
    # Query['lang']='*'
    # mongo_db = "twitter_db"
    # mongo_tweet_collection = 'تظاهرة الدراجات الجزائر'
    # mongo_user = f"AlgeriaTwitterGraph"
    # Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user)

   
    # query="MATCH (p:User{Checked: false})-[r:FOLLOWS]->({id_str:$id})RETURN p.id_str as id "
    # # Retrieve user IDs from Neo4j that hasn't been checked
    # Extractor.Graph_Extraction(mongo_db,mongo_user,query,verbose=True)




# cloud =CloudOfWord(mongo_uri,mongo_db,mongo_user,lang='french')
# cloud.print_Could()
# def Algerian_location(location):
#     Locations=[ 'Algérie','Algiers','Alger','Algeria','الجزائر','Algiers, Algeria']
#     if location in Locations:
#         return True
#     for loc in Locations:
#         if loc in location: return True
#     return False




# print(Algerian_location('Alger'))
# print(Algerian_location('skikda_Alger'))
# print(Algerian_location('skikda - Algérie'))
# print(Algerian_location('الجزائر - hsh'))
# print(Algerian_location('us - hsh'))
