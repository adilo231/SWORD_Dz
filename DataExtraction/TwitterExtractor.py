import tweepy
import time
from DataStorage.DBHandlers import  GraphDBHandler,DocDBHandler
import random
import pickle
import os
from tqdm import tqdm
import json

Locations=[ 'Algérie','Algiers','Alger','Algeria','الجزائر','dz','Tipaza','bladi','setif','Oran',
                'Adrar', 'أدرار', 'Chlef', 'الشلف', 'Laghouat', 'الأغواط', 'Oum El Bouaghi', 'أم البواقي', 
                'Batna', 'باتنة', 'Béjaïa', 'بجاية', 'Biskra', 'بسكرة', 'Béchar', 'بشار', 'Blida', 'البليدة', 
                'Bouira', 'البويرة', 'Tamanrasset', 'تمنراست', 'Tébessa', 'تبسة', 'Tlemcen', 'تلمسان', 'Tiaret',
                'تيارت', 'Tizi Ouzou', 'تيزي وزو', 'Alger', 'الجزائر', 'Djelfa', 'الجلفة', 'Jijel', 'جيجل', 
                'Sétif', 'سطيف', 'Saïda', 'سعيدة', 'Skikda', 'سكيكدة', 'Sidi Bel Abbès', 'سيدي بلعباس', 'Annaba', 
                'عنابة', 'Guelma', 'قالمة', 'Constantine', 'قسنطينة', 'Médéa', 'المدية', 'Mostaganem', 'مستغانم', 
                'M\'Sila', 'المسيلة', 'Mascara', 'معسكر', 'Ouargla', 'ورقلة', 'Oran', 'وهران', 'El Bayadh', 'البيض', 
                'Illizi', 'إليزي', 'Bordj Bou Arréridj', 'برج بوعريريج', 'Boumerdès', 'بومرداس', 'El Tarf', 
                'الطارف', 'Tindouf', 'تندوف', 'Tissemsilt', 'تيسمسيلت', 'El Oued', 'الوادي', 'Khenchela', 'خنشلة', 
                'Souk Ahras', 'سوق أهراس', 'Tipaza', 'تيبازة', 'Mila', 'ميلة', 'Aïn Defla', 'عين الدفلى', 'Naâma', 
                'النعامة', 'Aïn Témouchent', 'عين تموشنت', 'Ghardaïa', 'غرداية', 'Relizane', 'غليزان']
def Algerian_location(location):
    if location in Locations:
        return True
    for loc in Locations:
        if loc in location: return True
    return False


class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        if status_code == 420:
            return False

class TweetExtractor():
    def __init__(self,Creadits=0):
        print('Init Extractore')
        self.graphDB_Driver =GraphDBHandler()

        # connect to MongoDB
        self.DocGB_Driver = DocDBHandler()

        f = open('authentification.json')
        data= json.load(f)
        API_credentials=data['API_credentials'][Creadits]

        # authenticate with Twitter API
        self.auth = tweepy.OAuthHandler(API_credentials["consumer_key"], API_credentials["consumer_secret"])
        self.auth.set_access_token(API_credentials["access_key"], API_credentials["access_secret"])

    
        # create API object
        self.api = tweepy.API(self.auth,wait_on_rate_limit = True,wait_on_rate_limit_notify=True)
        

    def Topic_Tweet_Extraction(self,query,mongo_db,mongo_tweet_collection,mongo_user_collection,verbose=False):
        """
            Extract tweets from Twitter's API that contain a given query string and store them in a MongoDB database.
            
            Parameters:
            -----------
            query: dict
                A dictionary that specifies the search query. The dictionary must have a 'query' key that maps to a
                string representing the query to be searched.
            mongo_db: str
                The name of the MongoDB database where the tweets and users will be stored.
            mongo_tweet_collection: str
                The name of the MongoDB collection where the tweets will be stored.
            mongo_user_collection: str
                The name of the MongoDB collection where the users will be stored.
            verbose: bool, optional
                If True, print information about the tweets being extracted. Default is False.
            
            Returns:
            --------
            None
                This function does not return anything, but it stores the extracted tweets and users in the MongoDB
                database.
        """
        
        
        # Set Mong DB and collection
        db = self.DocGB_Driver.myclient[mongo_db]
        tweet_collection = db[mongo_tweet_collection]
        user_collection = db[mongo_user_collection]
        # search for tweets containing the phrase 
        Nbr_tweets=1
        if verbose:
            print('Extraction Started')
        try:
            print(query['query'])
            for tweet in tweepy.Cursor(self.api.search, q=query['query'],tweet_mode='extended').items():
                
                    Tweet = tweet._json
                    user = Tweet['user']
                    try:
                         if verbose:
                            print(f"\tAdding tweet {Nbr_tweets} ",Tweet['text'][:20])
                    except :
                         if verbose:
                            print(f"\tAdding tweet {Nbr_tweets} ",Tweet['full_text'][:20])

                        
                    Tweet['user'] = user['id_str']
                    # check if the tweet already exists in MongoDB
                    if not tweet_collection.find_one({"id_str": Tweet['id_str']}):
                        # insert the tweet into MongoDB
                       
                        tweet_collection.insert_one(Tweet)
                        Nbr_tweets+=1
                        # check if the user already exists in MongoDB
                        if not user_collection.find_one({"id_str": user['id_str']}):
                            # insert the user into MongoDB
                            user_collection.insert_one(user)
                        else:
                            if verbose:
                                print(f"\t\tUser { user['id_str']} Already added....")
                    else:
                        if verbose:
                            print(f"\t\tTweet {Tweet['id_str']} Already added....")
                        
                    # if verbose:
                    #         print(f"\tAdd to neo4j")
                    # if the tweet is a retweet, create a relationship between the retweet and the source tweet in Neo4j
                    # if hasattr(tweet, "retweeted_status"):
                    #     with self.graphDB_Driver.session as session:
                    #         session.run(
                    #             "MERGE (t1:Tweet {id: $tweet_id_str,id_str: $tweet_id_str, MongoColTweet:$mongoDB})"
                    #             "MERGE (t2:Tweet {id: $source_id_str,id_str: $source_id_str, MongoColTweet:$mongoDB})"
                    #             "MERGE (t1)-[:retweeted_from]->(t2)",
                    #             tweet_id_str=tweet.id_str, source_id_str=tweet.retweeted_status.id_str,mongoDB=mongo_tweet_collection
                    #         )
                    #     # create a relationship between the user and the tweet in Neo4j Retweet
                    #     with self.graphDB_Driver.session as session:
                    #         session.run(
                    #             "MERGE (u:User {id: $user_id_str,id_str: $user_id_str, MongoColTweet:$mongoDB})"
                    #             "MERGE (t:Tweet {id: $tweet_id_str,id_str: $tweet_id_str, MongoColTweet:$mongoDB})"
                    #             "MERGE (u)-[:retweeted]->(t)",
                    #             user_id_str=user['id_str'], tweet_id_str=Tweet['id_str'],mongoDB=mongo_tweet_collection
                    #         )
                    # else:
                    #     # create a relationship between the user and the tweet in Neo4j
                    #     with self.graphDB_Driver.session as session:
                    #         session.run(
                    #             "MERGE (u:User {id: $user_id_str,id_str: $user_id_str, MongoColTweet:$mongoDB})"
                    #             "MERGE (t:Tweet {id: $tweet_id_str,id_str: $tweet_id_str, MongoColTweet:$mongoDB})"
                    #             "MERGE (u)-[:tweeted]->(t)",
                    #             user_id_str=user['id_str'], tweet_id_str=Tweet['id_str'],mongoDB=mongo_tweet_collection
                    #         )
                    if verbose:
                            print(f"\tAdded to neo4j...")
        except tweepy.TweepError as e:
                if "429" in str(e):
                    print("Sleeping for 5 mins untile rate limit reset...")
                    time.sleep(60 * 5) # wait for 15 minutes
                    
                else:
                    print("An error occurred:", e)

    def UploadIdstoNeo4j(self,mongo_user_collection):
        root="Data/cache"
        
        files=os.listdir(root)
        while True:
            files=os.listdir(root)
            if len((files))>0:
                random.shuffle(files)
                file=files[0]
                if not file.startswith('.'):
                    print(file)
                    
                    user_id=file.split('_')[0]
                    type=file.split('_')[1]
                    with open(f'{root}/{file}', 'rb') as f:
                        ids = pickle.load(f)
                    print(user_id,type,)
                    if type=="friend":
                        with self.graphDB_Driver.driver.session() as session:
                            for friend_id in tqdm(ids):
                                    
                                    result = session.run("MATCH (a:User {id_str: $user_id}) "
                                            "MERGE (b:User {id_str: $friend_id , Checked: $checked}) "
                                            "MERGE (a)-[:FOLLOWS]->(b)", 
                                            user_id=(user_id), friend_id=str(friend_id),checked=False)

                    elif type=="follower":
                        with self.graphDB_Driver.driver.session() as session:
                            for follower_id in tqdm(ids):
                                
                                result = session.run("MATCH (a:User {id_str: $user_id}) "
                                                    "MERGE (b:User {id_str: $follower_id, Checked: $checked}) "
                                                    "MERGE (b)-[:FOLLOWS]->(a)", 
                                                    user_id=(user_id), follower_id=str(follower_id),checked=False)
                print("all Ids has been uploaded in the file: ",f'{root}/{file}', 'file remove it')
                os.remove(f'{root}/{file}') 
            else:
                 print("No data to be Uploaded sleeping for 5 mins")
                 time.sleep(60*5)               

    def Graph_Extraction(self,mongo_db,mongo_user_collection,query,verbose=False):
       
        # Set Mong DB and collection
        db = self.DocGB_Driver.myclient[mongo_db]
        user_collection = db[mongo_user_collection]

                
        
        result = self.graphDB_Driver.session.run(query,id="1239152434309738496")
        user_ids = [record["id"] for record in result]
        if verbose:
            print("Number of users to be checked: ",len(user_ids))
        random.shuffle(user_ids)
        
        # Iterate over user IDs
        for user_id in user_ids:
            
            try:
                # Retrieve user's  from Twitter API
                user = self.api.get_user(user_id)._json
                if not user_collection.find_one({"id_str": user['id_str']}):
                    user['_id'] = user['id']
                    user['mongoCol'] = mongo_user_collection
                    user_collection.insert_one(user)
                else:
                    
                    if verbose:
                        print('\tAlready Added',user['screen_name'],user['id'])
                
                Algerian =Algerian_location(user['location']) 
                
                if verbose:
                    print(f"Extracting user', {user['screen_name']},{user['id']} location: {user['location']} Is it: {Algerian} ")
                with self.graphDB_Driver.driver.session() as session:
                    session.run(""" MATCH (u:User {id_str: $user_id})
                            SET u.screen_name=$screen_name,  
                                u.followers_count= $followers_count,
                                u.friends_count= $friends_count,
                                u.location= $location,
                                u.Checked= $checked,
                                u.Algerian=$Algerian,
                                u.cursor_friends = COALESCE(u.cursor_friends, -1),
                                u.cursor_followers = COALESCE(u.cursor_followers, -1)
                               
                            """,
                            user_id=(user_id),
                            screen_name=user['screen_name'],
                            followers_count=user['followers_count'],
                            friends_count=user['friends_count'],
                            location=user['location'],
                            checked=False,
                            Algerian=Algerian,

                            )
                
                if (Algerian):
                    # Get Friends IDs
                    if verbose:
                        print(f"\t Extracting friends and followers of user_id: {user_id} of a location: {user['location']} ")
                        print(f"\t\t getting friend: {user['friends_count']}")
                        
                    with self.graphDB_Driver.driver.session() as session:
                        results=session.run("""  MATCH (n{id_str: $user_id}) RETURN n.cursor_friends """,user_id= str(user_id))
                        cursor=results.single()[0]

                    butch_number=1
                    friend_ids = []
                    
                    while cursor != 0 and user['friends_count']>0:
                        friend_ids = []
                        if verbose:
                            print(f"\t\t\t cursor {cursor}")


                        time.sleep(0.5)
                        resutls=tweepy.Cursor(self.api.friends_ids,cursor=cursor, user_id=user_id,count=500).pages()
                        page = next(resutls)
                        cursor = resutls.next_cursor
                        with self.graphDB_Driver.driver.session() as session:
                            session.run(""" Merge (u:User {id_str: $user_id})
                                            SET 
                                            u.cursor_friends=$cursor 
                                            RETURN u""",
                                            user_id= str(user_id),
                                            cursor=cursor
                                            )
                        friend_ids.extend(page)
                        if verbose:
                            print(f"\t\t\t cursor {cursor}")
                       
                        if verbose:
                            print(f"\t\t\t{butch_number}/{round(len(friend_ids)/500)+1} -> {len(friend_ids)} extracted ready to load to DB")
                        with open(f"Data/cache/{user_id}_friend_{butch_number}.pkl", 'wb') as f:
                                pickle.dump(friend_ids, f)
                        butch_number+=1
                        for friend_id in friend_ids:
                            try:
                                
                                with self.graphDB_Driver.driver.session() as session:
                                    result = session.run("MERGE (a:User {id_str: $user_id}) "
                                        "MERGE (b:User {id_str: $friend_id }) "
                                        "MERGE (a)-[:FOLLOWS]->(b)"
                                        "SET  b.Checked = COALESCE(b.Checked, false)", 
                                        user_id=str(user_id), friend_id=str(friend_id))

                            except tweepy.TweepError as e:
                                print(f"Error fetching friends/followers of user {user_id}: {str(e)}")
                                if "Rate limit exceeded" in str(e):
                                    print("Waiting for rate limit to reset...")
                                    time.sleep(60 * 15) # wait for 15 minutes
                        if verbose:
                            print(f"\t\t\t{len(friend_ids)} id friends has been added")
                    if verbose:
                                print(f"\t\t Extracting friend Ids end")           
                                print(f"\t\t Extracting followers Ids: {user['followers_count']}")
                    
                    
                    with self.graphDB_Driver.driver.session() as session:
                        results=session.run("""  MATCH (n{id_str: $user_id}) RETURN n.cursor_followers """,user_id= str(user_id))
                        cursor=results.single()[0]

                    
                    butch_number=1
                    # Get Followers IDs
                    while cursor != 0 and user['followers_count']>0:
                        follower_ids = []
                        if verbose:
                            print(f"\t\t\t cursor {cursor}")



                        resutls= tweepy.Cursor(self.api.followers_ids,cursor=cursor,  user_id=user_id,count=500).pages()
                        time.sleep(1)
                        page = next(resutls)
                        cursor = resutls.next_cursor
                        with self.graphDB_Driver.driver.session() as session:
                            session.run(""" Merge (u:User {id_str: $user_id})
                                            SET 
                                            u.cursor_followers=$cursor 
                                            RETURN u""",
                                            user_id= str(user_id),
                                            cursor=cursor
                                            )
                        follower_ids.extend(page)
                        if verbose:
                            print(f"\t\t\t cursor {cursor}")
                        
                        if verbose:
                            print(f"\t\t\t {butch_number}/{round(user['followers_count']/500)+1} -> {len(follower_ids)} ID extracted ready to load to DB")
                        with open(f"Data/cache/{user_id}_follower_{butch_number}.pkl", 'wb') as f:
                                pickle.dump(follower_ids, f)
                        butch_number+=1
                        for follower_id in follower_ids:
                            try:
                                  
                                
                                with self.graphDB_Driver.driver.session() as session:
                                    result = session.run("MERGE (a:User {id_str: $user_id}) "
                                                        "MERGE (b:User {id_str: $follower_id}) "
                                                        "MERGE (a)<-[:FOLLOWS]-(b)"
                                                        "SET  b.Checked = CASE WHEN b.Checked IS NULL THEN false ELSE b.Checked END", 
                                                        user_id=str(user_id), follower_id=str(follower_id))
                            except tweepy.TweepError as e:
                                print(f"Error fetching friends/followers of user {user_id}: {str(e)}")
                                if "429" in str(e):
                                    print("Waiting for rate limit to reset...")
                                    time.sleep(60 * 15) # wait for 15 minutes
                        if verbose:
                            print(f"\t\t\t{len(follower_ids)} ID extracted and has been added to BD")
                        if verbose:            
                                    print(f"\t\t Followers Ids extraction ends")
            except tweepy.TweepError as e:
                        print(f"Error fetching friends/followers of user {user_id}: {str(e)}")
                        if "429" in str(e):
                            print("Waiting for rate limit to reset...")
                            time.sleep(60 * 15) # wait for 15 minutes
            with self.graphDB_Driver.driver.session() as session:
                            session.run(""" Merge (u:User {id_str: $user_id})
                                            SET 
                                            u.Checked= $checked,
                                            u.followerExtractered=$followerExtractered 
                                            RETURN u""",
                                            user_id= str(user_id),
                                            checked=True,
                                            followerExtractered=True
                                            )


    def Get_Tweets_in_streaming(self,mongo_db,collection):
        db = self.DocGB_Driver.myclient[mongo_db]
        collection = db[collection]
        myStreamListener = MyStreamListener()
        myStream = tweepy.Stream(auth = self.auth, listener=myStreamListener)
        print('streaming')
        # Start streaming
        myStream.filter(track=['us', 'musk'])  # Filter tweets containing keywords
        print('streaming2')



