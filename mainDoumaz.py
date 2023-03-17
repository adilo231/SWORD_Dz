from Transformer.transformerrs import *

# Connect to MongoDB
client_mongo = "mongodb://localhost:27017/"
#
# Select the database and collection
db_name = "twitter_db"


collection_names=['Mahrez']


transforms=transform(db_name)

# #remove null attributes
# transforms.remove_and_update_null('Mahrez',verbose=True)


# # # add tokens to docs
# transforms.doc_update_tokens('Mahrez',verbose=True)

# #show clod of words of each collection
# transforms.cloud_of_words('Mahrez',verbose=True)

# #show lang distribution of each collection
transforms.tweets_lang_repartition('Mahrez',verbose=True)


#transform dates from string to datetime
transforms.string_to_datetime('Mahrez',verbose=True)


#show date distribution of each collection
#-------------------------------------------------------------------------------------------------------------#
#----------------------IL FAUT PASSER PAR LE TRANSFORMER string_to_datetime()---------------------------------#
#-------------------------------------------------------------------------------------------------------------#

transforms.plot_tweets_per_day('Mahrez',verbose=True)


plt.show()
