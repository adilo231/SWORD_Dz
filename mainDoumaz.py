from Transformer.transformerrs import *
from Transformer.TextPreprocessing import *

#
# Select the database and collection
db_name = "twitter_db"


collection_names='Mahrez'


transforms=transform(db_name)

# # pipline of transformers
# transforms.pipeline(collection_names,remove_null=True,cloud_words=True,lang_dist=True,date_dist=True,stance_dist=True,localisation_dist=True,verbose=True)

# transforms.stance_language_repartition(collection_names,verbose=True)
# #META dATA
# #NUMBER OF DOCS PER COLLECTION
# transforms.nbr_doc_per_collection()
# #META LANG REPARTITION
# transforms.meta_tweets_lang_repartition(verbose=True)

transforms.Topic_detection(collection_names,verbose=True)



