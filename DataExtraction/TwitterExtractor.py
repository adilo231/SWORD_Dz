import tweepy
from pymongo import MongoClient
import time
from DataExtraction.DBHandlers import  GraphDBHandler,DocDBHandler
from neo4j import GraphDatabase

class TweetExtractor():
    def __init__(self,connectionStringDocDB,neo_uri,userGDB,passwordGDB,API_credentials):
        print('Initn Extractore')
        self.graphDB_Driver =GraphDBHandler(neo_uri,userGDB,passwordGDB)

        # connect to MongoDB
        self.DocGB_Driver = DocDBHandler(connectionStringDocDB)

        # authenticate with Twitter API
        self.auth = tweepy.OAuthHandler(API_credentials["consumer_key"], API_credentials["consumer_secret"])
        self.auth.set_access_token(API_credentials["access_key"], API_credentials["access_secret"])

    
        # create API object
        self.api = tweepy.API(self.auth,wait_on_rate_limit = True,wait_on_rate_limit_notify=True)
        print('Init Extractore End')

    def Topic_Tweet_Extraction(self,query,mongo_db,mongo_tweet_collection,mongo_user_collection,verbose=False):
        # Set Mong DB and collection
        db = self.DocGB_Driver.myclient[mongo_db]
        tweet_collection = db[mongo_tweet_collection]
        user_collection = db[mongo_user_collection]
        # search for tweets containing the phrase 
        
        if verbose:
            print('Extraction Started')
        
        for tweet in tweepy.Cursor(self.api.search, q=query['query'],tweet_mode='extended').items():
            try:
                Tweet = tweet._json
                user = Tweet['user']
                Tweet['user'] = user['id_str']
                # check if the tweet already exists in MongoDB
                if not tweet_collection.find_one({"id_str": Tweet['id_str']}):
                    # insert the tweet into MongoDB
                    # tweet_collection.insert_one({"id_str": tweet.id_str, "user_id_str": tweet.user.id_str})
                    tweet_collection.insert_one(Tweet)
                    # check if the user already exists in MongoDB
                    if not user_collection.find_one({"id_str": user['id_str']}):
                        # insert the user into MongoDB
                        user_collection.insert_one(user)
                    

                # if the tweet is a retweet, create a relationship between the retweet and the source tweet in Neo4j
                if hasattr(tweet, "retweeted_status"):
                    with self.graphDB_Driver.session as session:
                        session.run(
                            "MERGE (t1:Tweet {id_str: $tweet_id_str, MongoCol:$mongoDB})"
                            "MERGE (t2:Tweet {id_str: $source_id_str, MongoCol:$mongoDB})"
                            "MERGE (t1)-[:retweeted_from]->(t2)",
                            tweet_id_str=tweet.id_str, source_id_str=tweet.retweeted_status.id_str,mongoDB=mongo_tweet_collection
                        )
                    # create a relationship between the user and the tweet in Neo4j Retweet
                    with self.graphDB_Driver.session as session:
                        session.run(
                            "MERGE (u:User {id_str: $user_id_str, MongoCol:$mongoDB})"
                            "MERGE (t:Tweet {id_str: $tweet_id_str, MongoCol:$mongoDB})"
                            "MERGE (u)-[:retweeted]->(t)",
                            user_id_str=user['id_str'], tweet_id_str=Tweet['id_str'],mongoDB=mongo_tweet_collection
                        )
                else:
                    # create a relationship between the user and the tweet in Neo4j
                    with self.graphDB_Driver.session as session:
                        session.run(
                            "MERGE (u:User {id_str: $user_id_str, MongoCol:$mongoDB})"
                            "MERGE (t:Tweet {id_str: $tweet_id_str, MongoCol:$mongoDB})"
                            "MERGE (u)-[:tweeted]->(t)",
                            user_id_str=user['id_str'], tweet_id_str=Tweet['id_str'],mongoDB=mongo_tweet_collection
                        )
            except tweepy.TweepError as e:
                if "429" in str(e):
                    print("Sleeping for 5 mins untile rate limit reset...")
                    time.sleep(60 * 5) # wait for 15 minutes
                    continue
                else:
                    print("An error occurred:", e)
                    continue

    def Graph_Extraction(self,mongo_db,mongo_user_collection,Locations):
         # Set Mong DB and collection
        db = self.DocGB_Driver.myclient[mongo_db]
        user_collection = db[mongo_user_collection]

                
        query = "MATCH (u:User {Checked: false}) RETURN u.id AS id"

        # Retrieve user IDs from Neo4j
        
        result = self.graphDB_Driver.session.run(query)
        user_ids = [record["id"] for record in result]
        print(len(user_ids))
        user_ids=user_ids[:5]
        print((user_ids))
        # Iterate over user IDs
        for user_id in user_ids:
            print(f'get folower of user {user_id}')
            try:
                # Retrieve user's  from Twitter API
                user = self.api.get_user(user_id)._json
                if not user_collection.find_one({"id_str": user['id_str']}):
                    user['_id'] = user['id']
                    user['mongoCol'] = mongo_user_collection
                    user_collection.insert_one(user)

                with self.graphDB_Driver.driver.session() as session:
                    session.run(""" MATCH (u:User {id: $user_id})
                            SET u.screen_name=$screen_name,  
                                u.id_str= $id_str,
                                u.followers_count= $followers_count,
                                u.location= $location,
                                u.friends_count= $friends_count,
                                u.MongoCol= $mongo_col,
                                u.Checked= $checked 
                            RETURN u""",
                            screen_name=user['screen_name'],
                            user_id=str(user['id']),
                            id_str=user['id_str'],
                            followers_count=user['followers_count'],
                            location=user['location'],
                            friends_count=user['friends_count'],
                            mongo_col=mongo_user_collection,
                            checked=True
                            )
                print(f" location {user['location']} {(user['location'] in Locations)} {user['followers_count']< 2000}")
                if ((user['location'] in Locations) and user['followers_count']< 2000 ):
                    # Get Friends IDs
                    print(f"getting friend and follower user {user_id} of a location {user['location']} follower {user['followers_count']} friend {user['friends_count']}")
                    for friend_id in tweepy.Cursor(self.api.friends_ids, user_id=user_id).items():
                        try:
                            friend_id = str(friend_id)
                            with self.graphDB_Driver.driver.session() as session:
                                result = session.run("MERGE (a:User {id: $user_id}) "
                                    "MERGE (b:User {id: $friend_id ,MongoCol: $mongo_col, Checked: $checked}) "
                                    "MERGE (a)-[:FOLLOWS]->(b)", 
                                    user_id=user_id, friend_id=friend_id,mongo_col=mongo_user_collection,checked=False)
                        except tweepy.TweepError as e:
                            print(f"Error fetching friends/followers of user {user_id}: {str(e)}")
                            if "Rate limit exceeded" in str(e):
                                print("Waiting for rate limit to reset...")
                                time.sleep(60 * 15) # wait for 15 minutes
                                


                    # Get Followers IDs
                    for follower_id in tweepy.Cursor(self.api.followers_ids, user_id=user_id).items():
                        try:
                            follower_id = str(follower_id)
                            with self.graphDB_Driver.driver.session() as session:
                                result = session.run("MERGE (a:User {id: $user_id}) "
                                                    "MERGE (b:User {id: $follower_id,MongoCol: $mongo_col, Checked: $checked}) "
                                                    "MERGE (b)-[:FOLLOWS]->(a)", 
                                                    user_id=user_id, follower_id=follower_id,mongo_col=mongo_user_collection,checked=False)
                        except tweepy.TweepError as e:
                            print(f"Error fetching friends/followers of user {user_id}: {str(e)}")
                            if "429" in str(e):
                                print("Waiting for rate limit to reset...")
                                time.sleep(60 * 15) # wait for 15 minutes
            except tweepy.TweepError as e:
                        print(f"Error fetching friends/followers of user {user_id}: {str(e)}")
                        if "429" in str(e):
                            print("Waiting for rate limit to reset...")
                            time.sleep(60 * 15) # wait for 15 minutes





