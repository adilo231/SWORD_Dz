import DataStorage.GraphGenerator as gg
from DataExtraction.TwitterExtractor import  TweetExtractor
import Model.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
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

    keywords = ['Algeria','Algérie','Alger','Algiers','الجزائر']

    # Define the search query
    query = " OR ".join(keywords)
    Extractor =TweetExtractor(mongo_uri,neo_uri,neo_user,neo_password,API_credentials)
    Query={
        'query' : query,


    }
    

    mongo_db = "twitter_db"
    mongo_tweet_collection = keywords[0]
    mongo_user_collection = f"{keywords[0]}_users"
    Extractor.Topic_Tweet_Extraction( Query,mongo_db,mongo_tweet_collection,mongo_user_collection)

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