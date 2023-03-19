import pymongo
import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import emoji
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import pandas as pd
from pymongo import MongoClient
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import matplotlib.pyplot as plt
from DataStorage.DBHandlers import  DocDBHandler

import arabic_reshaper
from bidi.algorithm import get_display

# nltk.download('wordnet')

from tqdm import tqdm
        
class transform :
    def __init__(self,db_name):
        # connect to MongoDB
        self.DocGB_Driver = DocDBHandler()
        
        self.db_name=db_name
        

    def remove_and_update_null(self,collection_name,verbose=False):
        
        if verbose:
            print("remove and update null attributes")
        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]
      
        tweets = db[collection_name]
        docs=tweets.find({})
        count = tweets.count_documents({})
        if verbose:
            print('\t Number of tweet containg null values ',count)
        for doc in docs:
            if 'updated'  not in doc.keys():
                update_dict = {}
                for key, value in doc.items():
                    if value != None:
                        update_dict[key] = value
                update_dict['updated'] = True
                tweets.replace_one({"_id": doc["_id"]}, update_dict)

    def text_to_tokens(self,text,verbose=False):
        if verbose:
            print("\t Removing all links")
        # Replace any URLs in the tweet with the string 'URL'
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'RT', '', text)
        if verbose:
            print("\t --> Text prepocessing")
        # preprocess the text
        tokens = nltk.word_tokenize(text.lower())
        tokens = [emoji.demojize(token) for token in tokens]
        tokens = [token for token in tokens if not any(c.isdigit() for c in token)]
        stop_words = set(nltk.corpus.stopwords.words()+ list(string.punctuation)+["'","’","”","“",",","،","¨","‘","؛","’","``","''",'’','“','”']+list(string.digits))
        words = [word for sent in tokens for (word, pos) in nltk.pos_tag(word_tokenize(sent)) if (pos not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']) and (word not in stop_words)]
        if verbose:
            print("\t --> Text lemmatize words")
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words
    
    def doc_update_tokens(self,collection_name,verbose=True):
        

        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]
        if verbose:
            print("Getting tweets tokens")

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

    def Wordcloud_language_generator(self,lang,collection_name,verbose=False):
        # Query the database for all tweets and their corresponding frensh language
        results = collection_name.find({"lang":lang})
        r=collection_name.count_documents({"lang":lang})
        words=[]
        if(r>0):    
            for result in results:
                words=words+result['tokens']
            if lang=='ar':
                arabic_pattern = r'^[\u0600-\u06FF]+$'
                words_reshaped=[]
                for word in words:
                        
                        # The above pattern matches any Unicode character in the Arabic range
                        # The ^ and $ characters are used to indicate that the whole string should be matched
        
                    if bool(re.match(arabic_pattern, word)):
                        reshaped_text = arabic_reshaper.reshape(word)
                        display_text = get_display(reshaped_text)
                        display_text = display_text.replace("_", "")
                        words_reshaped.append(display_text)
                # else:
                #     words_reshaped.append(word)
                if len(words_reshaped)==0:
                    words_reshaped.append("NoTweet")
                words=words_reshaped
            else:
                arabic_pattern = r'^[\u0600-\u06FF]+$'
                words_reshaped=[]
                for word in words:
                    word = word.replace("_", "")  
        
                    if bool(re.match(arabic_pattern, word))==False:
                        
                        words_reshaped.append(word)
                # else:
                #     words_reshaped.append(word)
                if len(words_reshaped)==0:
                    words_reshaped.append("NoTweet")
                words=words_reshaped

        else:
            words.append("NoTweet")
        
        
        # create a frequency distribution of the words
        freq_dist = nltk.FreqDist(words)

        # create a word cloud from the most frequent words
        # if len(words)<=1:
        #         wordcloud=None
        # else:
        if lang=="ar" :
            wordcloud = WordCloud(width=1600, height=800,font_path='font/NotoSansArabic_SemiCondensed-ExtraBold.ttf', background_color='white').generate_from_frequencies(freq_dist)
    
        else:
            wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(freq_dist)
        
        
        return wordcloud, words
       
    
    def string_to_datetime(self,collection_name,verbose=True):
        
        if verbose:
            print("Converting all String datas to datetime format")
        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]
      
        collection=db[str(collection_name)]
        docs=collection.find({})
        for doc in docs:
            date_format = '%a %b %d %H:%M:%S %z %Y'
            if'date' not in  doc.keys():
                datee= datetime.strptime(doc['created_at'], date_format)
                collection.update_one( {"_id": doc["_id"]} , { "$set": { "date": datee } })

    def cloud_of_words(self,collection_name, verbose=False):
       

        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]

        if verbose:
            print(f'Starting with collection {collection_name}')

        collection = db[str(collection_name)]
        # Query the database for all tweets and their corresponding languages
        wordcloud_fr, word_fr = self.Wordcloud_language_generator(lang="fr", collection_name=collection,
                                                                verbose=verbose)
        wordcloud_ar, word_ar = self.Wordcloud_language_generator(lang="ar", collection_name=collection,
                                                                verbose=verbose)
        wordcloud_en, word_en = self.Wordcloud_language_generator(lang="en", collection_name=collection,
                                                                verbose=verbose)

        # Create a new figure for each collection
        i=int(len(word_fr)>1)+int(len(word_ar)>1)+int(len(word_en)>1)
        fig, axs = plt.subplots(figsize=(20, 8), nrows=1,ncols=i)
        print(i)
        j=0
        

        if(len(word_fr)>1):
            #print(len(word_fr),j)
            if(i>1):
                axs[j].imshow(wordcloud_fr)
                axs[j].set_title(f'Collection Cloud of word {collection_name} in French')
            else:
                plt.imshow(wordcloud_fr)
                plt.title(f'Collection Cloud of word {collection_name} in French')
                plt.axis('off')
            j=j+1
        
        if(len(word_ar)>1):
            
            #print(len(word_ar),j)
            if(i>1):
                axs[j].imshow(wordcloud_ar)
                axs[j].set_title(f'Collection Cloud of word {collection_name} in Arabic')
            else:
                plt.imshow(wordcloud_ar)
                plt.title(f'Cloud of word {collection_name} in Arabic')
                plt.axis('off')
            j=j+1
        
        if(len(word_en)>1):
            #print(len(word_en),j)
            if(i>1):
                axs[j].imshow(wordcloud_en)
                axs[j].set_title(f'Cloud of word {collection_name} in English')
            else:
                plt.imshow(wordcloud_en)
                plt.title(f'Cloud of word {collection_name} in English')
                plt.axis('off')
        
        if(i>1):
            for ax in axs:
                ax.axis('off')

        # Set the title of the entire figure
        fig.suptitle(f'Word Clouds for Collection {collection_name}')



    
    def tweets_lang_repartition(self,collection_name, verbose):
        """
        Plot the language distribution of tweets for collection in MongoDB.

        Args:
        - self: instance of the class.
        - verbose (bool): If True, display progress of the execution. 

        Returns:
        - None.
        """
        
        if verbose:
            print("Getting tweets language distributions")
        
 

        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]



      
        
        # Select the collection
        collection = db[collection_name]
        fr_collection = int(collection.count_documents({"lang": "fr"}))
        ar_collection = int(collection.count_documents({"lang": "ar"}))
        ang_collection = int(collection.count_documents({"lang": "en"}))
        other_collection = int(collection.count_documents({})) - (fr_collection + ar_collection + ang_collection)

        plt.figure(figsize=(10,10))
        plt.style.use('ggplot') 
        plt.bar(['French', 'Arabic', 'English', 'Other'], [fr_collection, ar_collection, ang_collection, other_collection])
        plt.title("Distribution of number of tweets by language")

        # Add numbers to bars
        for i, v in enumerate([fr_collection, ar_collection, ang_collection, other_collection]):
            plt.text(i, v + 100, str(v), ha='center')
       

        
     


        # Display the figure
        if verbose:
            print("Displaying the language distributions of tweets")
        

        

    def plot_tweets_per_day(self,collection_name, verbose=False):
        """
        Plot the number of tweets per day for each collection in the database and save each plot as a PNG file.
                Parameters:
        -----------
        verbose: bool, optional
            If True, display information about each step of the function. Default is False.
        """
        date_format = '%a %b %d %H:%M:%S %z %Y'

        

        # Select the database
        db = self.DocGB_Driver.myclient[self.db_name]
        if verbose:
            print("Selected database:", self.db_name)

        i=0
       
        # Select the collection
        tweets = db[collection_name]
        
        # Query the database for all tweets and their corresponding dates
        results = tweets.find({}).sort('date', pymongo.ASCENDING)
        count = tweets.count_documents({})
        if verbose:
            print(f"Number of documents in {collection_name}: {count}")

        # Get the first and last tweet dates
        first_date =results[0]
        last_date = results[count -1]
        if verbose:
            print(f"First tweet date: {first_date['created_at']}")
            print(f"Last tweet date: {last_date['created_at']}")
        
        # Calculate the number of tweets per day
        tweet_counts = {}
        first_day = min(datetime.strptime(first_date['created_at'], date_format).date(),
                        datetime.strptime(last_date['created_at'], date_format).date())
        last_day  = max(datetime.strptime(first_date['created_at'], date_format).date(),
                        datetime.strptime(last_date['created_at'], date_format).date())
        
        tweets = db[collection_name]
        results = tweets.find({}).sort('date', pymongo.ASCENDING)
        for result in results:
            tweet_date = datetime.strptime(result['created_at'], date_format).date()
            if first_day <= tweet_date <= last_day:
                current_date_str = tweet_date
                if current_date_str not in tweet_counts:
                    tweet_counts[current_date_str] = 0
                tweet_counts[current_date_str] += 1
        
        plt.style.use('ggplot')  
        # Create a new figure for each collection
        fig = plt.figure(figsize=(10, 6))
        # Plot the data as a bar chart in the new figure
        plt.plot(tweet_counts.keys(), tweet_counts.values(),'-bo')
        plt.title(collection_name)
        # Save the figure with a filename based on the collection name
        #fig.savefig(f"{collection_name}.png")
        if verbose:
            print(f"Figure {i+1} saved as {collection_name}.png")
        i += 1
