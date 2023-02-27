from Transformer.Could_of_word import CloudOfWord

mongo_db = "twitter_db"
mongo_user = "Prix"
mongo_uri = "mongodb://localhost:27017/"

cloud =CloudOfWord(mongo_uri,mongo_db,mongo_user,lang='french')
cloud.print_Could()