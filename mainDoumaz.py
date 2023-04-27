from Transformer.transformerrs import *
if __name__=="__main__":
    # Connect to MongoDB
    client_mongo = "mongodb://localhost:27017/"
    
    # Select the database and collection
    db_name = "twitter_db"


    collection_names="test"
    transforms=transform(db_name)


    # pipline of transformers
    transforms.pipeline(collection_names,remove_null=False,cloud_words=False,lang_dist=False,date_dist=False,stance_dist=False,localisation_dist=False,Topic_detection=True)


#     transforms.stance_language_repartition(collection_names,verbose=True)
#     # #META dATA
#     # #NUMBER OF DOCS PER COLLECTION
#     # transforms.nbr_doc_per_collection()
#     # #META LANG REPARTITION
#     # transforms.meta_tweets_lang_repartition(verbose=True)



#     plt.show()
