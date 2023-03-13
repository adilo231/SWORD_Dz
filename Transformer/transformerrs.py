import pymongo
import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import emoji
import string
#import connection_mongo
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from datetime import datetime,timedelta
from argparse import RawDescriptionHelpFormatter
import json
import os
import pandas as pd
from pymongo import MongoClient
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
import matplotlib.pyplot as plt
import math
        
class transform :
    def __init__(self) -> None:
        pass

    def remove_and_update_null(self,collection_names,db_name,mongo_client_url):
        client = pymongo.MongoClient(mongo_client_url)

        # Select the database and collection
        db = client[db_name]
        for  collection_name in collection_names:
            tweets = db[collection_name]
            docs=tweets.find({})
            count = tweets.count_documents({})
            print(count)
            for doc in docs:
                if 'updated'  not in doc.keys():
                    update_dict = {}
                    for key, value in doc.items():
                        if value != None:
                            update_dict[key] = value
                    update_dict['updated'] = True
                    tweets.replace_one({"_id": doc["_id"]}, update_dict)

    def text_to_tokens(self,text):
        # Replace any URLs in the tweet with the string 'URL'
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'RT', '', text)

        # preprocess the text
        tokens = nltk.word_tokenize(text.lower())
        tokens = [emoji.demojize(token) for token in tokens]
        tokens = [token for token in tokens if not any(c.isdigit() for c in token)]
        stop_words = set(nltk.corpus.stopwords.words()+ list(string.punctuation)+["'","’","”","“",",","،","¨","‘","؛","’","``","''",'’','“','”']+list(string.digits))
        words = [word for sent in tokens for (word, pos) in nltk.pos_tag(word_tokenize(sent)) if (pos not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']) and (word not in stop_words)]
        
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words
    
    def doc_update_tokens(self,collection_names,db_name,mongo_client_url):
        client = pymongo.MongoClient(mongo_client_url)

        # Select the database and collection
        db = client[db_name]
        for  collection_name in collection_names:
            print(collection_name)
            collection=db[str(collection_name)]
            docs=collection.find({})
            for doc in docs:
                if 'tokens'  not in doc.keys():
                    if 'full_text' in doc.keys():
                        text = doc['full_text'] 
                    else:
                        text = doc['text']
                    
                    words=self.text_to_tokens(text)    
                    

                    collection.update_one({"_id": doc["_id"]}, { "$set": { "tokens": words } })

    def Wordcloud_language_generator(self,lang,collection_name):
        # Query the database for all tweets and their corresponding frensh language
        results = collection_name.find({"lang":lang})
        r=collection_name.count_documents({"lang":lang})
        words=[]
        if(r>0):    
            for result in results:
                words=words+result['tokens']
        else:
            words.append("NoTweet")

        # create a frequency distribution of the words
        freq_dist = nltk.FreqDist(words)

        # create a word cloud from the most frequent words

        wordcloud = WordCloud(width=1600, height=800,font_path='font\kawkab-light.ttf', background_color='white').generate_from_frequencies(freq_dist)
        return wordcloud
    
    def string_to_datetime(self,collection_names,db_name,mongo_client_url):
        client = pymongo.MongoClient(mongo_client_url)

        # Select the database and collection
        db = client[db_name]
        for  collection_name in collection_names:
            collection=db[str(collection_name)]
            docs=collection.find({})
            for doc in docs:
                date_format = '%a %b %d %H:%M:%S %z %Y'
                if'date' not in  doc.keys():
                    datee= datetime.strptime(doc['created_at'], date_format)
                    collection.update_one( {"_id": doc["_id"]} , { "$set": { "date": datee } })

    def cloud_of_words(self,collection_names,db_name,mongo_client_url):
        # Connect to MongoDB "mongodb://localhost:27017/"
        client = pymongo.MongoClient(mongo_client_url)

        # Select the database and collection
        db = client[db_name]
        #collection_names=['test','Morocco Algeria']
        #collection_names=['Hirake','CDC','Fertility','Morocco Algeria','Prix','Teboune','Algeria','Ukraine']

        # Create the grid of subplots
        fig, axs = plt.subplots(figsize=(20, 8), ncols=len(collection_names), nrows=3)

        # Flatten the axs array so that we can iterate over it with a single loop
        axs = axs.flatten()

        for i, collection_name in enumerate(collection_names):
            print(collection_name, "start")
            collection=db[str(collection_name)]
            wordcloud_fr=self.Wordcloud_language_generator(lang="fr",collection_name=collection)
            print("french done")

            # Query the database for all tweets and their corresponding arabic language
            wordcloud_ar=self.Wordcloud_language_generator(lang="ar",collection_name=collection)
            print("arabic done")

            # Query the database for all tweets and their corresponding english language
            wordcloud_en=self.Wordcloud_language_generator(lang="en",collection_name=collection)
            print("english done")
            # plot the word cloud in a subplot
            axs[i].imshow(wordcloud_fr)
            axs[i].set_title(str(collection_name)+str('_French'))

            axs[i+ len(collection_names)].imshow(wordcloud_ar)
            axs[i+ len(collection_names)].set_title(str(collection_name)+str('_Arabic'))

            axs[i+2*len(collection_names)].imshow(wordcloud_en)
            axs[i+2*len(collection_names)].set_title(str(collection_name)+str('_English'))

        for ax in axs:
            ax.axis('off')

        # Set the title of the entire figure
        fig.suptitle('Word Clouds for Different Languages')

        #plt.show()



    def tweets_lang_repartition(self, collection_names, db_name, mongo_client_url):
        # Connect to the MongoDB server
        Stat_Global = pd.DataFrame()
        client = MongoClient(mongo_client_url)

        # Select the database and collection
        db = client[db_name]

        num_collections = len(collection_names)
        rows = int(num_collections / 2) + num_collections % 2
        cols = 2 if num_collections > 1 else 1

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 6 * rows), squeeze=False)


        for i, collection_name in enumerate(collection_names):
            # Select the collection
            collection = db[collection_name]
            fr_collection = int(collection.count_documents({"lang": "fr"}))
            ar_collection = int(collection.count_documents({"lang": "ar"}))
            ang_collection = int(collection.count_documents({"lang": "en"}))
            other_collection = int(collection.count_documents({})) - (fr_collection + ar_collection + ang_collection)

            # Plot the language repartition for the current collection
            row = i // cols
            col = i % cols
            axs[row, col].bar(['French', 'Arabic', 'English', 'Other'], [fr_collection, ar_collection, ang_collection, other_collection])
            axs[row, col].set_title(collection_name)

        # adjust the spacing between the subplots
        plt.subplots_adjust(hspace=0.5)
        # Set the title of the entire figure
        fig.suptitle('lang distribution of each collection')

        # display the figure
        #plt.show()

    

    def tweets_date_repartition(self,collection_names, db_name, mongo_client_url):
        date_format = '%a %b %d %H:%M:%S %z %Y'

        # Connect to MongoDB
        client = pymongo.MongoClient(mongo_client_url)

        # Select the database and collection
        db = client[db_name]

        # Calculate the number of rows and columns for the grid of subplots
        n_cols = min(2, len(collection_names))
        n_rows = (len(collection_names) - 1) // 2 + 1

        # Create the grid of subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 8))
        fig.subplots_adjust(bottom=0.05, top=0.9, wspace=0.3, hspace=0.3)

        # Flatten the axs array so that we can iterate over it with a single loop
        if(len(collection_names)>1):
            axs = axs.flatten()

        for i, collection_name in enumerate(collection_names):
            # Select the collection
            tweets = db[collection_name]
            
            # Query the database for all tweets and their corresponding dates
            results = tweets.find({}).sort('date',pymongo.ASCENDING)
            count = tweets.count_documents({})

            # Get the first and last tweet dates
            first_date = results[0]
            last_date = results[count - 1]

            # Calculate the number of tweets per day
            tweet_counts = {}
            first_day = min(datetime.strptime(first_date['created_at'], date_format).date(),datetime.strptime(last_date['created_at'], date_format).date())
            last_day = max(datetime.strptime(first_date['created_at'], date_format).date(),datetime.strptime(last_date['created_at'], date_format).date())

            current_date = first_day
            while current_date <= last_day:
                current_date_str = current_date
                tweet_counts[current_date_str] = 0
                tweets = db[collection_name]
                result = tweets.find({}).sort('date',pymongo.ASCENDING)

                for date in result:
                    tweet_date = datetime.strptime(date['created_at'], date_format).date()
                    if tweet_date == current_date :
                        tweet_counts[current_date_str] += 1

                current_date += timedelta(days=1)

            # Plot the data as a bar chart in the appropriate subplot
            if(len(collection_names)>1):
                axs[i].plot(tweet_counts.keys(), tweet_counts.values(), '-bo')
                axs[i].set_title(collection_name)

        # Hide any unused subplots
        for i in range(len(collection_names), n_rows * n_cols):
            axs[i].axis('off')

        # Set the title of the entire figure
        fig.suptitle('Date distribution of each collection')
        
        # Display the plot
        #plt.show()
