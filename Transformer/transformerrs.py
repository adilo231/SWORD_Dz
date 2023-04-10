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
from DataStorage.DBHandlers import DocDBHandler
import pickle

import arabic_reshaper
from bidi.algorithm import get_display


from tqdm import tqdm
import csv
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentIntensityAnalyzer


# nltk.download('wordnet')

from tqdm import tqdm
import os


class transform:
    def __init__(self, db_name):
        # connect to MongoDB
        self.DocGB_Driver = DocDBHandler()

        self.db_name = db_name

        if os.path.exists('Transformer/Classifier/ArabicClassifier.pkl'):
            print(" Arabic Classifier exist")
            with open('Transformer/Classifier/ArabicClassifier.pkl', 'rb') as f:
                    self.ArabicClassifier = pickle.load(f)
        else:
            print("Arabic Classifier does not exist")
            self.ArabicClassifier=self.Train_arabic_classifier()


        if os.path.exists('Transformer/Classifier/FrenchClassifier.pkl'):
            print("French Classifier exist")
            with open('Transformer/Classifier/FrenchClassifier.pkl', 'rb') as f:
                    self.FrenchClassifier = pickle.load(f)
        else:
            print(" French Classifier does not exist")
            self.FrenchClassifier=self.Train_French_classifier()

        if os.path.exists('Transformer/Classifier/EnglishClassifier.pkl'):
            print("English Classifier exist")
            with open('Transformer/Classifier/EnglishClassifier.pkl', 'rb') as f:
                    self.EnglishClassifier = pickle.load(f)
        else:
            print("English Classifier does not exist")
            self.EnglishClassifier=self.Train_English_classifier() 
        

    def remove_and_update_null(self, collection_name, verbose=False):

        if verbose:
            print("remove and update null attributes")
        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]
        
        collection = db[str(collection_name)]
        
       
        docs = collection.find({})
        count = collection.count_documents({})
        updated=1
        for doc in docs:
            if'updated' in doc.keys():
                updated+=1
        null_arg_docs= count - updated
        if verbose:
            print('\t Number of tweet containg null values ', null_arg_docs)
        if(null_arg_docs != 0):
            
            docs = collection.find({})
            for doc in tqdm(docs,desc="removing null attributes"):
                if 'updated' not in doc.keys() and doc['_id']!='metadata':
                    
                    update_dict = {}
                    for key, value in doc.items():
                        if value != None:
                            update_dict[key] = value
                    update_dict['updated'] = True
                    collection.replace_one({"_id": doc["_id"]}, update_dict)
                    
            doc=collection.find_one({'_id': 'metadata'})
        
            collection.update_one({"_id": doc["_id"]}, {
                                        "$set": {"remove_null_update":count - 1 }})
        else:

            doc=collection.find_one({'_id': 'metadata'})
        
            collection.update_one({"_id": doc["_id"]}, {
                                        "$set": {"remove_null_update":count - 1 }})

    def text_to_tokens(self, text, verbose=False):
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
        tokens = [token for token in tokens if not any(
            c.isdigit() for c in token)]
        stop_words = set(nltk.corpus.stopwords.words() + list(string.punctuation)+[
                         "'", "’", "”", "“", ",", "،", "¨", "‘", "؛", "’", "``", "''", '’', '“', '”']+list(string.digits))
        words = [word for sent in tokens for (word, pos) in nltk.pos_tag(word_tokenize(sent)) if (pos not in [
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']) and (word not in stop_words)]
        if verbose:
            print("\t --> Text lemmatize words")
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words

    def doc_update_tokens(self, collection_name, verbose=True):

        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]
        if verbose:
            print("Getting tweets tokens")

        collection = db[str(collection_name)]
        docs = collection.find({})
        # count=collection.count_documents({})
        meta=collection.find_one({'_id': 'metadata'})
        if(meta['doc_tokens_update']!=meta['number_of_document'] ):
            for doc in tqdm(docs,desc="tokenise docs"):
                if doc['_id']!='metadata' :
                    if 'tokens' not in doc.keys():
                        if 'full_text' in doc.keys():
                            text = doc['full_text']
                        else:
                            text = doc['text']

                        words = self.text_to_tokens(text)
                        collection.update_one({"_id": doc["_id"]}, {
                                            "$set": {"tokens": words}})
            collection.update_one({"_id": meta["_id"]}, {
                                        "$set": {"doc_tokens_update":meta['number_of_document'] }})

    def Wordcloud_language_generator(self, lang, collection_name, verbose=False):
        # Query the database for all tweets and their corresponding frensh language
        results = collection_name.find({"lang": lang})
        r = collection_name.count_documents({"lang": lang})
        words = []
        if(r > 0):
            print("Collecting tokens")
            for result in results:
                if 'tokens' in result.keys():
                    words = words+result['tokens']
            print("filtering tokens")

            if lang == 'ar':
                arabic_pattern = r'^[\u0600-\u06FF]+$'
                words_reshaped = []
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
                if len(words_reshaped) == 0:
                    words_reshaped.append("NoTweet")
                words = words_reshaped
            else:
                arabic_pattern = r'^[\u0600-\u06FF]+$'
                words_reshaped = []
                for word in words:
                    word = word.replace("_", "")

                    if bool(re.match(arabic_pattern, word)) == False:

                        words_reshaped.append(word)
                # else:
                #     words_reshaped.append(word)
                if len(words_reshaped) == 0:
                    words_reshaped.append("NoTweet")
                words = words_reshaped

        else:
            words.append("NoTweet")

        # create a frequency distribution of the words
        freq_dist = nltk.FreqDist(words)

        # create a word cloud from the most frequent words
        # if len(words)<=1:
        #         wordcloud=None
        # else:
        print("generating cloud of words")
        if lang == "ar":
            wordcloud = WordCloud(width=1600, height=800, font_path='font/NotoSansArabic_SemiCondensed-ExtraBold.ttf',
                                  background_color='white').generate_from_frequencies(freq_dist)

        else:
            wordcloud = WordCloud(
                width=1600, height=800, background_color='white').generate_from_frequencies(freq_dist)

        return wordcloud, words,freq_dist

    def string_to_datetime(self, collection_name, verbose=True):

        if verbose:
            print("Converting all String datas to datetime format")
        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]

        collection = db[str(collection_name)]
        docs = collection.find({})
        for doc in tqdm(docs):
            date_format = '%a %b %d %H:%M:%S %z %Y'
            if'date' not in doc.keys() and doc['_id']!='metadata':
                datee = datetime.strptime(doc['created_at'], date_format)
                collection.update_one({"_id": doc["_id"]}, {
                                      "$set": {"date": datee}})

    def cloud_of_words(self, collection_name, verbose=False):

        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]

        if verbose:
            print(f'Starting with collection {collection_name}')

        collection = db[str(collection_name)]

        meta=collection.find_one({'_id': 'metadata'})
        ar_count = collection.count_documents({"lang": "ar"})
        fr_count = collection.count_documents({"lang": "fr"})
        en_count = collection.count_documents({"lang": "en"})

        somme= ar_count + fr_count+ en_count
        if(meta['cloud_words_update']==somme):
            freq_dist_ar=meta['arabic_cloud_words_update']
            freq_dist_fr=meta['frensh_cloud_words_update']
            freq_dist_en=meta['english_cloud_words_update']
            word_ar=meta['word_arabic']
            word_fr=meta['word_frensh']
            word_en=meta['word_english']

            wordcloud_ar = WordCloud(width=1600, height=800, font_path='font/NotoSansArabic_SemiCondensed-ExtraBold.ttf',
                                  background_color='white').generate_from_frequencies(freq_dist_ar)

            wordcloud_fr = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(freq_dist_fr)
            wordcloud_en = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(freq_dist_en)

        else:    
            # Query the database for all tweets and their corresponding languages
            print("French CW")
            wordcloud_fr, word_fr,freq_dist_fr = self.Wordcloud_language_generator(lang="fr", collection_name=collection,
                                                                    verbose=verbose)
            print("English CW")
            wordcloud_en, word_en ,freq_dist_en= self.Wordcloud_language_generator(lang="en", collection_name=collection,
                                                                    verbose=verbose)

            print("Arabic CW")
            wordcloud_ar, word_ar,freq_dist_ar = self.Wordcloud_language_generator(lang="ar", collection_name=collection,
                                                                    verbose=verbose)
            collection.update_one({"_id": meta["_id"]}, {
                                        "$set": {"cloud_words_update":somme,"arabic_cloud_words_update":freq_dist_ar,'word_arabic':word_ar,'frensh_cloud_words_update':freq_dist_fr,'word_frensh':word_fr,'english_cloud_words_update':freq_dist_en,'word_english':word_en }})
            
        # Create a new figure for each collection
        i = int(len(word_fr) > 1)+int(len(word_ar) > 1)+int(len(word_en) > 1)
        fig, axs = plt.subplots(figsize=(20, 8), nrows=1, ncols=i)
        print(i)
        j = 0

        if(len(word_fr) > 1):
            # print(len(word_fr),j)
            if(i > 1):
                axs[j].imshow(wordcloud_fr)
                axs[j].set_title(
                    f'Collection Cloud of word {collection_name} in French')
            else:
                plt.imshow(wordcloud_fr)
                plt.title(
                    f'Collection Cloud of word {collection_name} in French')
                plt.axis('off')
            j = j+1

        if(len(word_ar) > 1):

            # print(len(word_ar),j)
            if(i > 1):
                axs[j].imshow(wordcloud_ar)
                axs[j].set_title(
                    f'Collection Cloud of word {collection_name} in Arabic')
            else:
                plt.imshow(wordcloud_ar)
                plt.title(f'Cloud of word {collection_name} in Arabic')
                plt.axis('off')
            j = j+1

        if(len(word_en) > 1):
            # print(len(word_en),j)
            if(i > 1):
                axs[j].imshow(wordcloud_en)
                axs[j].set_title(f'Cloud of word {collection_name} in English')
            else:
                plt.imshow(wordcloud_en)
                plt.title(f'Cloud of word {collection_name} in English')
                plt.axis('off')

        if(i > 1):
            for ax in axs:
                ax.axis('off')

        # Set the title of the entire figure
        fig.suptitle(f'Word Clouds for Collection {collection_name}')

    def tweets_lang_repartition(self, collection_name, verbose):
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
        other_collection = int(collection.count_documents({})) - (fr_collection + ar_collection + ang_collection)-1

        meta=collection.find_one({'_id': 'metadata'})
        
        collection.update_one({"_id": meta["_id"]}, {
                                        "$set": {"doc_lang_dist_update":int(collection.count_documents({}))-1,'arabic_language_documents':ar_collection,'french_language_documents':fr_collection,'english_language_documents': ang_collection,'other_language_documents':other_collection}})



        plt.figure(figsize=(10, 10))
        plt.style.use('ggplot')
        plt.bar(['French', 'Arabic', 'English', 'Other'], [
                fr_collection, ar_collection, ang_collection, other_collection])
        plt.title("Distribution of number of tweets by language")

        # Add numbers to bars
        for j, v in enumerate([fr_collection, ar_collection, ang_collection, other_collection]):
            plt.text(j, v + 10, str(v), ha='center')

        # Display the figure
        if verbose:
            print("Displaying the language distributions of tweets")

    def plot_tweets_per_day(self, collection_name, verbose=False):
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

        i = 0

        # Select the collection
        tweets = db[collection_name]

        # Query the database for all tweets and their corresponding dates
        results = tweets.find({}).sort('date', pymongo.ASCENDING)
        count = tweets.count_documents({})-1
        if verbose:
            print(f"Number of documents in {collection_name}: {count}")

        # Get the first and last tweet dates
        first_date = results[1]
        last_date = results[count - 1]
        if verbose:
            print(f"First tweet date: {first_date['created_at']}")
            print(f"Last tweet date: {last_date['created_at']}")

        # Calculate the number of tweets per day
        tweet_counts = {}
        first_day = min(datetime.strptime(first_date['created_at'], date_format).date(),
                        datetime.strptime(last_date['created_at'], date_format).date())
        last_day = max(datetime.strptime(first_date['created_at'], date_format).date(),
                       datetime.strptime(last_date['created_at'], date_format).date())

        tweets = db[collection_name]
        results = tweets.find({}).sort('date', pymongo.ASCENDING)
        for result in results:
            if(result['_id']!='metadata'):
                tweet_date = datetime.strptime(
                    result['created_at'], date_format).date()
                if first_day <= tweet_date <= last_day:
                    current_date_str = tweet_date
                    if current_date_str not in tweet_counts:
                        tweet_counts[current_date_str] = 0
                    tweet_counts[current_date_str] += 1
        tweet_list = [f"{date.strftime('%Y-%m-%d')}: {count}" for date, count in tweet_counts.items()]
        print(tweet_list)
        meta=tweets.find_one({'_id': 'metadata'})
        
        tweets.update_one({"_id": meta["_id"]}, {
                                        "$set": {"doc_date_dist_update":count,'number_of_doc_per-day':tweet_list }})
        plt.style.use('ggplot')
        # Create a new figure for each collection
        fig = plt.figure(figsize=(10, 6))
        # Plot the data as a bar chart in the new figure
        plt.plot(tweet_counts.keys(), tweet_counts.values(), '-bo')
        plt.title(collection_name)
        # Save the figure with a filename based on the collection name
        # fig.savefig(f"{collection_name}.png")
        # if verbose:
        #     print(f"Figure {i+1} saved as {collection_name}.png")
        i += 1

    def extract_features(self, tokens, word_features, verbose=False):
        token_set = set(tokens)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in token_set)
        return features
    
    def Train_arabic_classifier(self):
        # preparing Classifier
            print("\t preparing Arabic Classifier", "\n")
            data = []
            # Open the CSV file
            with open('Transformer/Classifier/train_all_ext.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row if there is one

                # Loop through each row in the CSV file
                for row in reader:
                    # Extract the text and stance from the row
                    text = row[2]  # Assuming the text is in the first column
                    # Assuming the stance is in the second column
                    stance = row[1]

                    # Append the (text, stance) pair to the data list
                    data.append((text, stance))

            stop_words = set(stopwords.words('arabic'))
            stemmer = SnowballStemmer('arabic')

            preprocessed_data = []
            for text, stance in data:
                tokens = word_tokenize(text)
                filtered_tokens = [stemmer.stem(
                    token) for token in tokens if token not in stop_words]
                preprocessed_data.append((filtered_tokens, stance))

            # Extract features
            all_words = nltk.FreqDist(
                [token for text, stance in preprocessed_data for token in text])
            word_features = list(all_words)[:1000]

            featuresets = [(self.extract_features(text, word_features), stance)
                           for (text, stance) in preprocessed_data]

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(
                featuresets, test_size=0.2, random_state=42)

            # Train the classifier
            classifier = NaiveBayesClassifier.train(train_set)
                        # Evaluate the classifier
            accuracy = nltk.classify.accuracy(classifier, test_set)
            print("Accuracy:", accuracy)
            with open('Transformer/Classifier/ArabicClassifier.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            return classifier

    def Train_French_classifier(self):
      # preparing Classifier
            print("preparing French Classifier", "\n")
            data = []
            # Open the CSV file
            with open('Transformer/Classifier/betsentiment-FR-tweets-sentiment-teams.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row if there is one

                # Loop through each row in the CSV file
                for row in reader:
                    # Extract the text and stance from the row
                    text = row[2]  # Assuming the text is in the first column
                    # Assuming the stance is in the second column
                    stance = row[4]

                    # Append the (text, stance) pair to the data list
                    data.append((text, stance))

            stop_words = set(stopwords.words('french'))
            stemmer = SnowballStemmer('french')
            preprocessed_data = []
            for text, stance in data:
                tokens = word_tokenize(text)
                filtered_tokens = [stemmer.stem(
                    token) for token in tokens if token not in stop_words]
                preprocessed_data.append((filtered_tokens, stance))

            # Extract features
            all_words = nltk.FreqDist(
                [token for text, stance in preprocessed_data for token in text])
            word_features = list(all_words)[:1000]

            featuresets = [(self.extract_features(text, word_features), stance)
                           for (text, stance) in preprocessed_data]

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(
                featuresets, test_size=0.2, random_state=42)

            # Train the classifier
            classifier = NaiveBayesClassifier.train(train_set)

            # Evaluate the classifier
            # accuracy = nltk.classify.accuracy(classifier, test_set)
            # print("Accuracy:", accuracy)
            with open('Transformer/Classifier/FrenchClassifier.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            return classifier


    def Train_English_classifier(self):
          # preparing Classifier
            print("preparing  English Classifier", "\n")
            # Download the pre-trained sentiment analyzer
            nltk.download('vader_lexicon')

            # Initialize the sentiment analyzer
            sid = SentimentIntensityAnalyzer()
            with open('Transformer/Classifier/EnglishClassifier.pkl', 'wb') as f:
                pickle.dump(sid, f)
            return sid

    def arabic_stance_classification(self,collection):
        
        # classification

        docs = collection.find({"lang": "ar"})
        ar_positif = 0
        ar_negatif = 0
        ar_neutre = 0
        i = 0
    
        if os.path.exists('Transformer/Classifier/data_ar.pkl'):
            with open('Transformer/Classifier/data_ar.pkl', 'rb') as f:
                    preprocessed_data = pickle.load(f)
                    print("exttract done")
        else:
            
            with open('Transformer/Classifier/train_all_ext.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row if there is one
                data=[]
                # Loop through each row in the CSV file
                for row in reader:
                    # Extract the text and stance from the row
                    text = row[2]  # Assuming the text is in the first column
                    # Assuming the stance is in the second column
                    stance = row[1]

                    # Append the (text, stance) pair to the data list
                    data.append((text, stance))
                    
                stop_words = set(stopwords.words('arabic'))
                stemmer = SnowballStemmer('arabic')
                preprocessed_data = []
                for text, stance in tqdm(data,desc="stemmatisation ar"):
                    tokens = word_tokenize(text)
                    filtered_tokens = [stemmer.stem(
                        token) for token in tokens if token not in stop_words]
                    preprocessed_data.append((filtered_tokens, stance))

            with open('Transformer/Classifier/data_ar.pkl', 'wb') as f:
                        # print("data ar pkl ")
                        pickle.dump(preprocessed_data, f)
                        print("data ar pkl  done ")



        
        all_words = nltk.FreqDist(
            [token for text, stance in preprocessed_data for token in text])
        word_features = list(all_words)[:1000]

        for doc in tqdm(docs):

            if'stance' in doc.keys():
                stance=doc['stance']
            else:
                if'text' in doc.keys():
                    tokens = word_tokenize(doc['text'])
                if'full_text' in doc.keys():
                    tokens = word_tokenize(doc['full_text'])
                stop_words = set(stopwords.words('arabic'))
                stemmer = SnowballStemmer('arabic')

                filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

                # Extraire les features du texte prétraité
                features = self.extract_features(filtered_tokens, word_features)

                # Classer le texte en utilisant le classificateur entraîné
                stance = self.ArabicClassifier.classify(features)
                collection.update_one({"_id": doc["_id"]}, {
                                "$set": {"stance": stance}})

            if stance == 'neutral':
                ar_neutre = ar_neutre + 1
            if stance == 'negative':
                ar_negatif = ar_negatif + 1
            if stance == 'positive':
                ar_positif = ar_positif + 1
            # Afficher le stance prédit
            #print("Stance prédit [",i,"] : ", stance)
            # i=i+1
        #print("+ - +-",ar_positif, ar_negatif,  ar_neutre)
        return ar_positif, ar_negatif, ar_neutre
    
    def french_stance_classification(self,collection):
        # preparing Classifier
        if os.path.exists('Transformer/Classifier/data_fr.pkl'):
            with open('Transformer/Classifier/data_fr.pkl', 'rb') as f:
                    preprocessed_data = pickle.load(f)
                    print("exttract done")

        else:
            data = []
            # Open the CSV file
            with open('Transformer/Classifier/betsentiment-FR-tweets-sentiment-teams.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row if there is one

                # Loop through each row in the CSV file
                for row in reader:
                    # Extract the text and stance from the row
                    text = row[2]  # Assuming the text is in the first column
                    # Assuming the stance is in the second column
                    stance = row[4]

                    # Append the (text, stance) pair to the data list
                    data.append((text, stance))
                stop_words = set(stopwords.words('french'))
                stemmer = SnowballStemmer('french')
                preprocessed_data = []
                for text, stance in tqdm(data,desc="stemmatisation fr"):
                    tokens = word_tokenize(text)
                    filtered_tokens = [stemmer.stem(
                        token) for token in tokens if token not in stop_words]
                    preprocessed_data.append((filtered_tokens, stance))
            with open('Transformer/Classifier/data_fr.pkl', 'wb') as f:
                        pickle.dump(preprocessed_data, f)

        

        # Extract features
        all_words = nltk.FreqDist(
            [token for text, stance in preprocessed_data for token in text])
        word_features = list(all_words)[:1000]

        
        # classification

        docs = collection.find({"lang": "fr"})
        fr_positif = 0
        fr_negatif = 0
        fr_neutre = 0
        i = 0
        for doc in tqdm(docs):
            if'stance' in doc.keys():
                stance=doc['stance']
            else:
                if'text' in doc.keys():
                    tokens = word_tokenize(doc['text'])
                if'full_text' in doc.keys():
                    tokens = word_tokenize(doc['full_text'])
                stop_words = set(stopwords.words('french'))
                stemmer = SnowballStemmer('french')

                filtered_tokens = [stemmer.stem(
                    token) for token in tokens if token not in stop_words]

                # Extraire les features du texte prétraité
                features = self.extract_features(
                    filtered_tokens, word_features)

                # Classer le texte en utilisant le classificateur entraîné
                stance = self.FrenchClassifier.classify(features)
                collection.update_one({"_id": doc["_id"]}, {
                                    "$set": {"stance": stance}})
            if stance == 'NEUTRAL':
                fr_neutre = fr_neutre + 1
            if stance == 'NEGATIVE':
                fr_negatif = fr_negatif + 1
            if stance == 'POSITIVE':
                fr_positif = fr_positif + 1
            # Afficher le stance prédit
            #print("Stance prédit [",i,"] : ", stance)
            # i=i+1
        #print("+ - +-",fr_positif, fr_negatif,  fr_neutre)
        return fr_positif, fr_negatif,  fr_neutre
    
    def english_stance_classification(self,collection):
        docs = collection.find({"lang": "en"})
        en_positif = 0
        en_negatif = 0
        en_neutre = 0
        

        for doc in tqdm(docs):
            # Example text to classify
            if'stance' in doc.keys():
                sentiment=doc['stance']
            else:
                if 'text' in doc.keys():
                    text = doc['text']
                if 'full_text' in doc.keys():
                    text = doc['full_text']

                # Classify the text
                scores = self.EnglishClassifier.polarity_scores(text)
                    
                # Determine the overall sentiment
                if scores['compound'] > 0:
                    sentiment = 'positive'
                    # en_positif = en_positif + 1

                elif scores['compound'] < 0:
                    sentiment = 'negative'
                    # en_negatif = en_negatif + 1

                else:
                    sentiment = 'neutral'
                    # en_neutre = en_neutre + 1
                collection.update_one({"_id": doc["_id"]}, {
                                    "$set": {"stance": sentiment}})
        
            if sentiment == 'neutral':
                en_neutre = en_neutre + 1
            if sentiment == 'negative':
                en_negatif = en_negatif + 1
            if sentiment == 'positive':
                en_positif = en_positif + 1
        # Print the sentiment
        #print("+ - +-",en_positif,en_negatif ,en_neutre)
        return en_positif,en_negatif ,en_neutre
    
    def stance_language_repartition(self, collection_name, verbose=False):

        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]

        if verbose:
            print(f'Starting with collection {collection_name}')

        collection = db[str(collection_name)]

        meta=collection.find_one({'_id': 'metadata'})
        
       
      
        ar_count = collection.count_documents({"lang": "ar"})
        fr_count = collection.count_documents({"lang": "fr"})
        en_count = collection.count_documents({"lang": "en"})
        
        number_graphs = 0
        if verbose:
            print("ar_count", ar_count)
            print("fr_count", fr_count)
            print("en_count", en_count)
        if(meta['arabic_stance']!=ar_count and meta['french_stance']!=fr_count and meta['english_stance']!=en_count ):
            if ar_count > 0:
                number_graphs += 1
                ar_positif, ar_negatif, ar_neutre=self.arabic_stance_classification(collection=collection)

                
            if fr_count > 0:
                number_graphs = number_graphs + 1
                
                fr_positif, fr_negatif,  fr_neutre=self.french_stance_classification(collection=collection)
            if en_count > 0:
                number_graphs = number_graphs + 1
                en_positif,en_negatif ,en_neutre =self.english_stance_classification(collection=collection)
            
            collection.update_one({"_id": meta["_id"]}, {
                                        "$set": {"doc_stance_dist_update":ar_count+fr_count+en_count,
                                                 'arabic_stance':ar_count,'arabic_stance_positif':ar_positif,'arabic_stance_negatif':ar_negatif,'arabic_stance_neutre':ar_neutre,
                                                 'french_stance':fr_count,'french_stance_positif':fr_positif,'french_stance_negatif':fr_negatif,'french_stance_neutre':fr_neutre,
                                                 'english_stance':en_count,'english_stance_positif':en_positif,'english_stance_negatif':en_negatif,'english_stance_neutre':en_neutre }})
        
        else:
            if ar_count > 0:
                number_graphs +=1
                ar_positif=meta['arabic_stance_positif']
                ar_negatif=meta['arabic_stance_negatif']
                ar_neutre=meta['arabic_stance_neutre']
            if fr_count > 0:
                number_graphs +=1
                fr_positif=meta['french_stance_positif']
                fr_negatif=meta['french_stance_negatif']
                fr_neutre=meta['french_stance_neutre']
            if en_count > 0:
                number_graphs +=1
                en_positif=meta['english_stance_positif']
                en_negatif=meta['english_stance_negatif']
                en_neutre=meta['english_stance_neutre']
        # print(number_graphs)
        # Plot
        fig, axs = plt.subplots(1, number_graphs, figsize=(10, 5))
        i = 0
        # Arabic subplot
        if ar_count > 0:
            ar_data = [ar_positif, ar_negatif, ar_neutre]
            if number_graphs == 1:
                plt.bar(['Positive', 'Negative', 'Neutral'], ar_data)
                plt.title(str(collection_name)+' Arabic Stance')
                for j, v in enumerate([ar_positif, ar_negatif, ar_neutre]):
                    plt.text(j, v + 10, str(v), ha='center')
            else:
                axs[i].bar(['Positive', 'Negative', 'Neutral'], ar_data)
                axs[i].set_title(str(collection_name)+' Arabic Stance')
                for j, v in enumerate([ar_positif, ar_negatif, ar_neutre]):
                    axs[i].text(j, v + 10, str(v), ha='center')
                i = i+1

        if fr_count > 0:
            # French subplot
            fr_data = [fr_positif, fr_negatif, fr_neutre]
            if number_graphs == 1:
                plt.bar(['Positive', 'Negative', 'Neutral'], ar_data)
                plt.title(str(collection_name)+' French Stance')
                # Add numbers to bars
                for j, v in enumerate([fr_positif, fr_negatif, fr_neutre]):
                    plt.text(j, v + 10, str(v), ha='center')
            else:
                axs[i].bar(['Positive', 'Negative', 'Neutral'], fr_data)
                axs[i].set_title(str(collection_name)+' French Stance')
                # Add numbers to bars
                for j, v in enumerate([fr_positif, fr_negatif, fr_neutre]):
                    axs[i].text(j, v + 10, str(v), ha='center')
                i = i+1

        if en_count > 0:
            # English subplot
            en_data = [en_positif, en_negatif, en_neutre]
            if number_graphs == 1:
                plt.bar(['Positive', 'Negative', 'Neutral'], ar_data)
                plt.title(str(collection_name)+' English Stance')
                for j, v in enumerate([en_positif, en_negatif, en_neutre]):
                    plt.text(j, v + 10, str(v), ha='center')
            else:
                axs[i].bar(['Positive', 'Negative', 'Neutral'], en_data)
                axs[i].set_title(str(collection_name)+' English Stance')
                for j, v in enumerate([en_positif, en_negatif, en_neutre]):
                    axs[i].text(j, v + 10, str(v), ha='center')
                    
        plt.style.use('ggplot')
        # plt.show()


    def nbr_doc_per_collection(self):

        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]
        # Get the names of all collections in the database
        collection_names = db.list_collection_names()

        # Initialize lists to store collection names and document counts
        names = []
        counts = []

        # Loop through each collection and count the number of documents
        for collection_name in collection_names:
            collection = db[collection_name]
            num_docs = collection.count_documents({})
            names.append(collection_name)
            counts.append(num_docs)

        # Create a horizontal bar chart of the document counts for each collection
        plt.barh(names, counts)
        plt.title("Number of documents per collection")
        plt.xlabel("Number of documents")
        plt.ylabel("Collection name")
        plt.style.use('ggplot')
        for j, v in enumerate(counts):
            plt.text(v + 10000, j, str(v), ha='center')
        # plt.show()
    def meta_tweets_lang_repartition(self, verbose=False):
        """
        Plot the language distribution of tweets for all collections in MongoDB.

        Args:
        - self: instance of the class.
        - verbose (bool): If True, display progress of the execution. 

        Returns:
        - None.
        """
        if verbose:
            print("Getting tweets language distributions")

        # Select the database
        db = self.DocGB_Driver.myclient[self.db_name]

        # Get the names of all collections in the database
        collection_names = db.list_collection_names()

        # Initialize counters for French, English, and Arabic documents
        fr_count = 0
        en_count = 0
        ar_count = 0
        other_count=0
        
        # Loop through each collection and count the number of documents for each language
        for collection_name in collection_names:
            collection = db[collection_name]
            fr_count += collection.count_documents({"lang": "fr"})
            en_count += collection.count_documents({"lang": "en"})
            ar_count += collection.count_documents({"lang": "ar"})
            other= collection.count_documents({})- collection.count_documents({"lang": "fr"}) - collection.count_documents({"lang": "ar"}) -collection.count_documents({"lang": "en"})
            other_count += other
        # # Calculate the count for Other language documents
        # other_count = db.tweets.count_documents(
        #     {"$and": [{"lang": {"$ne": "fr"}}, {"lang": {"$ne": "en"}}, {"lang": {"$ne": "ar"}}]})

        # Create a bar chart of the document counts for each language
        plt.figure(figsize=(10, 10))
        plt.style.use('ggplot')
        plt.bar(['French', 'Arabic', 'English', 'Other'], [
                fr_count, ar_count, en_count, other_count])
        plt.title("Distribution of number of tweets by language of all collections")

        # Add numbers to bars
        for j, v in enumerate([fr_count, ar_count, en_count, other_count]):
            plt.text(j, v + 10, str(v), ha='center')

        # Display the figure
        if verbose:
            print("Displaying the language distributions of tweets")
        # plt.show()

    def localisation_distribution(self, collection_name, verbose=False):
         # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]

        

        collection = db[str(collection_name)]
        
        count=collection.count_documents({})-1
        meta=collection.find_one({'_id': 'metadata'})
        if (meta['number_of_document']!= count or meta['doc_localisation_dist']==0):
            if verbose:
                print(f'Starting localisation with collection {collection_name}')

            # Get the names of the "user_collections" collections
            user_collections = ["AlgeriaTwitterGraph", "International_users"]

            # Step 1: Récupérer les utilisateurs uniques de la collection "test"
            users = collection.distinct("user")

            # Step 2: Récupérer tous les documents de chaque collection de "user_collections" qui correspondent aux utilisateurs de "test",
            # et stocker les résultats dans une liste
            user_docs = []
            for user_collection_name in user_collections:
                user_collection = db[user_collection_name]
                docs = user_collection.find({ "id_str": { "$in": users } })
                user_docs += list(docs)

            # Step 3: Extraire les utilisateurs uniques des documents récupérés dans l'étape 2
            users_in_user_docs = list(set([doc["id_str"] for doc in user_docs]))

            # Step 4: Récupérer tous les documents de la collection "test" qui correspondent aux utilisateurs récupérés dans l'étape 3
            test_docs = collection.find({ "user": { "$in": users_in_user_docs } })

            # Step 5: Extraire les emplacements uniques des documents récupérés dans l'étape 2
            locations = list(set([doc["location"] for doc in user_docs]))

            # Step 6: Pour chaque emplacement, calculer le nombre de documents correspondant dans les collections "AlgeriaTwitterGraph" et "International_users"
            counts = {}
            for location in locations:
                count = 0
                for user_collection_name in user_collections:
                    user_collection = db[user_collection_name]
                    count += user_collection.count_documents({ "location": location, "id_str": { "$in": users_in_user_docs } })
                counts[location] = count

            # Remove the first item from the counts dictionary
            del counts[''] 
            

            # Step 7: Trier les emplacements par nombre de documents correspondants
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

            # Step 8: Conserver les 20 premiers emplacements les plus apparus
            top_locations = dict(sorted_counts[:20])

            cleaned_location_dict = {}

            for key, value in top_locations.items():
                    new_key = emoji.demojize(key)
                    cleaned_location_dict[new_key] = value

            for k, v in top_locations.items():
                k = k.strip()  # remove leading/trailing white space
                k = " ".join(k.split())  # replace multiple white space with single space
                cleaned_location_dict[k] = v
            
            top_locations=cleaned_location_dict   
            collection.update_one({"_id": meta["_id"]}, {
                                        "$set": {"doc_localisation_dist":top_locations}})
            
            # print(cleaned_location_dict)
        else:
            top_locations=meta['doc_localisation_dist']
        # print(top_locations)
        # Create a bar chart of the results with swapped axes
        plt.style.use('ggplot')
        plt.barh(range(len(top_locations)), list(top_locations.values()), align='center')
        plt.yticks(range(len(top_locations)), list(top_locations.keys()))

        # Set the plot title and axis labels
        plt.title('Nombre de documents par emplacement')
        plt.xlabel('Nombre de documents')
        plt.ylabel('Emplacement')
        
        # Show the plot
        # plt.show()
    
    def create_metadoc(self, collection_name, verbose=False):
        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]

        
        collection = db[str(collection_name)]

        nbr_doc=collection.count_documents({})
        doc_find=collection.count_documents({'_id': 'metadata'})
    
        # Create a document with _id set to "metadata"
        remove_null_update=0
        doc_tokens_update=0
        cloud_words_update=0
        arabic_cloud_words_update=0
        frensh_cloud_words_update=0
        english_cloud_words_update=0
        word_arabic=0
        word_frensh=0
        word_english=0
        doc_lang_dist_update=0
        ar_collection=0
        fr_collection=0
        ang_collection=0
        other_collection=0
        doc_date_dist_update=0
        tweet_date_counts=0
        doc_stance_dist_update=0
        ar_count=0
        ar_positif=0
        ar_negatif=0
        ar_neutre=0
        fr_count=0
        fr_positif=0
        fr_negatif=0
        fr_neutre=0
        en_count=0
        en_positif=0
        en_negatif=0
        en_neutre=0
        doc_localisation_dist=0

        document = { '_id': 'metadata', 'number_of_document': nbr_doc ,
                    'remove_null_update':remove_null_update,
                    'doc_tokens_update':doc_tokens_update,
                    'cloud_words_update':cloud_words_update,'arabic_cloud_words_update':arabic_cloud_words_update,'word_arabic':word_arabic,'frensh_cloud_words_update':frensh_cloud_words_update,'word_frensh':word_frensh,'english_cloud_words_update':english_cloud_words_update,'word_english':word_english,
                    'doc_lang_dist_update':doc_lang_dist_update,'arabic_language_documents':ar_collection,'french_language_documents':fr_collection,'english_language_documents': ang_collection,'other_language_documents':other_collection,
                    'doc_date_dist_update':doc_date_dist_update,'number_of_doc_per-day':tweet_date_counts,
                    'doc_stance_dist_update':doc_stance_dist_update,
                                                'arabic_stance':ar_count,'arabic_stance_positif':ar_positif,'arabic_stance_negatif':ar_negatif,'arabic_stance_neutre':ar_neutre,
                                                'french_stance':fr_count,'french_stance_positif':fr_positif,'french_stance_negatif':fr_negatif,'french_stance_neutre':fr_neutre,
                                                'english_stance':en_count,'english_stance_positif':en_positif,'english_stance_negatif':en_negatif,'english_stance_neutre':en_neutre,
                    'doc_localisation_dist':doc_localisation_dist }
        if(doc_find == 0):
            # Insert the document into the collection
            collection.insert_one(document)
        else:
            doc=collection.find_one({'_id': 'metadata'})
            if doc['number_of_document']!= collection.count_documents({}):
                doc['number_of_document']= collection.count_documents({})-1

            for key, value in document.items():
                if key not in doc:
                    doc[key] = value

            # Mise à jour du document existant avec les champs manquants
            collection.update_one({'_id': 'metadata'}, {'$set': doc})

    def pipeline(self,collection_name,remove_null=False,cloud_words=False,lang_dist=False,date_dist=False,stance_dist=False,localisation_dist=False,verbose=False):
        
        # create or update meta data document
        self.create_metadoc(collection_name, verbose=True) 

        #remove null arguments from documents
        if(remove_null==True):
             self.remove_and_update_null(collection_name,verbose=True)
        

        #number of tweets per localisation

        if(localisation_dist==True):
            self.localisation_distribution(collection_name,verbose)

        
        #WordCloud generator
        if cloud_words==True:
            #update documents with adding tokens
            self.doc_update_tokens(collection_name,verbose)
            #generation cloud of words
            self.cloud_of_words(collection_name,verbose)

        #number of tweets per language
        if(lang_dist==True):
            self.tweets_lang_repartition(collection_name,verbose)
        
        #number of tweets per date
        if(date_dist==True):
            self.string_to_datetime(collection_name,verbose)
            self.plot_tweets_per_day(collection_name,verbose)

       
        #stance repartition
        if(stance_dist==True):
            self.stance_language_repartition(collection_name,verbose)

        

        #show figures
        plt.show()
