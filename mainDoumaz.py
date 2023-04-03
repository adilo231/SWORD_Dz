from Transformer.transformerrs import *

# Connect to MongoDB
client_mongo = "mongodb://localhost:27017/"
#
# Select the database and collection
db_name = "twitter_db"


collection_names='Attaf'


transforms=transform(db_name)

    # pipline of transformers
    transforms.pipeline('logement',remove_null=False,cloud_words=False,lang_dist=False,date_dist=False,stance_dist=False,localisation_dist=True)


    # #META dATA
    # #NUMBER OF DOCS PER COLLECTION
    # transforms.nbr_doc_per_collection()
    # #META LANG REPARTITION
    # transforms.meta_tweets_lang_repartition(verbose=True)

    

    plt.show()
