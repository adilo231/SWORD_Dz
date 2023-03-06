
from DataExtraction.TwitterExtractor import  TweetExtractor
# Twitter API credentials
API_credentials={
'consumer_key' : "NlCb4KrL6LlB8HVB3RGYLy0lU",
'consumer_secret' : "GvACIXrd7jWvNWnIg7geb8gfL80j0L7OuTmmHCLl5DAUMDLwHb",
'access_key' : "1512829229356011530-CTLalKDfZPkJDPI2EfmG3CebbMXfDh",
'access_secret' : "w8ndvJ0FMK2qyzwzjqRBZ2ca57PAcho4QC8v2Xty6lQ35"}



# MongoDB credentials
mongo_uri = "mongodb://localhost:27017/"

# Neo4j credentials
neo_uri = "bolt://localhost:7687"
neo_user = "neo4j"
neo_password = "admin"






# define rate limit handler function
if __name__ == '__main__':
    Locations=[ 'Algérie','Algiers','Alger','Algeria','الجزائر','Alger-Algérie','Algiers, Algeria']
    mongo_db = "twitter_db"
    mongo_user = "AlgeriaTwitterGraph"
    Extractor =TweetExtractor(mongo_uri,neo_uri,neo_user,neo_password,API_credentials)
    Extractor.Graph_Extraction(mongo_db,mongo_user,Locations)
    # keywordslist=[]
    # keywordslist.append(["covid", "vaccines", "covid"])
    
    # Extractor =TweetExtractor(mongo_uri,neo_uri,neo_user,neo_password,API_credentials)

    # for keywords in keywordslist:
    # # Define the search query
    #     query = " AND  ".join(keywords)
        
    #     Query={
    #         'query' : query,
    #     }
    


    #     mongo_db = "twitter_db"
    #     mongo_tweet_collection = keywords[0]
    #     mongo_user_collection = f"CovidTweets_users"
    #     Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user_collection)

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