from Transformer.transformerrs import *

# Connect to MongoDB
client_mongo = "mongodb://localhost:27017/"
#
# Select the database and collection
db_name = "twitter_db"


collection_names=['DATT-DZ']


transforms=transform(collection_names=collection_names,db_name=db_name,mongo_client_url=client_mongo)

#remove null attributes
transforms.remove_and_update_null(verbose=True)
print('\n')

# add tokens to docs
transforms.doc_update_tokens()
print('\n')
#show clod of words of each collection
transforms.cloud_of_words()
print('\n')
#show lang distribution of each collection
transforms.tweets_lang_repartition()
print('\n')

#transform dates from string to datetime
transforms.string_to_datetime()
print('\n')

#show date distribution of each collection
#-------------------------------------------------------------------------------------------------------------#
#----------------------IL FAUT PASSER PAR LE TRANSFORMER string_to_datetime()---------------------------------#
#-------------------------------------------------------------------------------------------------------------#

transforms.tweets_date_repartition()


plt.show()
