from Transformer.transformerrs import *

# Connect to MongoDB
client_mongo = "mongodb://localhost:27017/"
#
# Select the database and collection
db_name = "twitter_db"


collection_names=['Prix']

transforms=transform()

#remove null attributes
transforms.remove_and_update_null(collection_names=collection_names,db_name=db_name,mongo_client_url=client_mongo,verbose=True)

#transform dates from string to datetime
transforms.string_to_datetime(collection_names=collection_names,db_name=db_name,mongo_client_url=client_mongo)

# add tokens to docs
transforms.doc_update_tokens(collection_names=collection_names,db_name=db_name,mongo_client_url=client_mongo)

#show clod of words of each collection
transforms.cloud_of_words(collection_names=collection_names,db_name=db_name,mongo_client_url=client_mongo)

# #show lang distribution of each collection
# transforms.tweets_lang_repartition(collection_names=collection_names,db_name=db_name,mongo_client_url=client_mongo)

# #show date distribution of each collection
# transforms.tweets_date_repartition(collection_names=collection_names,db_name=db_name,mongo_client_url=client_mongo)


plt.show()
