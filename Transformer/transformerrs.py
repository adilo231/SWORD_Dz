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
from pymongo import UpdateOne

import arabic_reshaper
from bidi.algorithm import get_display

import numpy as np
from tqdm import tqdm
import csv
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from Transformer.TextPreprocessing import *

# nltk.download('wordnet')

from tqdm import tqdm
import os


class transform:
    def __init__(self, db_name,verbose=False):
        # connect to MongoDB
        self.DocGB_Driver = DocDBHandler()

        self.db_name = db_name

        if os.path.exists('Transformer/Classifier/ArabicClassifier.pkl'):
            print(" Arabic Classifier exist")
            if verbose:
                print(" Arabic Classifier exist Loading...")
            with open('Transformer/Classifier/ArabicClassifier.pkl', 'rb') as f:
                    self.ArabicClassifier = pickle.load(f)
        else:
            print("Arabic Classifier does not exist")
            if verbose:
                print("Arabic classifier does not exist, creating classifier ")
            self.ArabicClassifier=self.__Train_arabic_classifier()


        if os.path.exists('Transformer/Classifier/FrenchClassifier.pkl'):
            print("French Classifier exist")
            if verbose:
                print("French Classifier exist")
            with open('Transformer/Classifier/FrenchClassifier.pkl', 'rb') as f:
                    self.FrenchClassifier = pickle.load(f)
        else:
            print(" French Classifier does not exist")
            if verbose:
                print(" French Classifier does not exist")
            self.FrenchClassifier=self.__Train_French_classifier()

        if os.path.exists('Transformer/Classifier/EnglishClassifier.pkl'):
            print("English Classifier exist")
            if verbose:
                print("English Classifier exist")
            with open('Transformer/Classifier/EnglishClassifier.pkl', 'rb') as f:
                    self.EnglishClassifier = pickle.load(f)
        else:
            print("English Classifier does not exist")
            if verbose:
                print("English Classifier does not exist")
            self.EnglishClassifier=self.__Train_English_classifier() 


    def __remove_and_update_null(self, collection_name, verbose=False):
        """
        The function updates the original document with the new dictionary using the replace_one method. 
        It also updates the "remove_null_update" key in the metadata document to reflect the number of documents that have been modified.

        If there are no documents containing null values,
          the function simply updates the "remove_null_update" key in the metadata document 
          with the total count of documents in the collection minus one.
        The verbose flag can be used to print additional information to the console during execution.
        """
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

    def __text_to_tokens(self, text, verbose=False):
        """
        This is a method in a Python class that takes in a string of text and preprocesses it by removing URLs and emojis,
            tokenizing the text into words, removing stop words and non-noun, non-adjective words, and lemmatizing the remaining words.
        
        
        The resulting list of words is returned as the output of the method. The method takes an optional verbose argument that,
            when set to True, prints out information about the preprocessing steps being taken. 
        
        The preprocessing steps include regular expression substitution, word tokenization, stop word removal, and lemmatization. 
        
        The method makes use of the nltk library for natural language processing and the WordNetLemmatizer class for word lemmatization.
        """

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

    def __doc_update_tokens(self, collection_name, verbose=True):
        """
        This is a method that updates the tokens for the documents in a MongoDB collection. 
        
        It first selects the collection and finds all the documents in it.
         
        Then it checks whether the tokens have already been updated for all the documents by comparing the metadata information.

        If the tokens haven't been updated, it uses the __text_to_tokens method to preprocess and tokenize the text in each document.
         
        Finally, it updates the collection with the new tokens and updates the metadata information to reflect the update.
        
        The method uses tqdm to display a progress bar while updating the tokens.
        
        """
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

                        words = self.__text_to_tokens(text)
                        collection.update_one({"_id": doc["_id"]}, {
                                            "$set": {"tokens": words}})
            collection.update_one({"_id": meta["_id"]}, {
                                        "$set": {"doc_tokens_update":meta['number_of_document'] }})

    def __Wordcloud_language_generator(self, lang, collection_name, verbose=False):

        """
        This is a method that generates a word cloud image based on the text data in a specific language from a MongoDB database.

        The method takes two arguments: a language code (e.g., "en" for English, "ar" for Arabic) and a MongoDB collection name.

        The method queries the MongoDB database for all documents that have the specified language code.

        If there are any documents in the database with the specified language code, the method collects all the text data 
            from those documents and stores them in a list called "words."

        The method then filters the words list based on whether the language is Arabic or not.
             If it's Arabic, the method uses an Arabic reshaper to properly display the Arabic text in the word cloud.
             If it's not Arabic, the method removes any Arabic words from the list.

        The method creates a frequency distribution of the words using the NLTK library.

        The method generates a word cloud image from the most frequent words using the WordCloud library. 
            If the language is Arabic, the method uses a specific Arabic font to properly display the Arabic text.

        The method returns the word cloud image, the list of words used to generate the word cloud, 
            and the frequency distribution of the words.
        """
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

    def __string_to_datetime(self, collection_name, verbose=True):
        """
        This is a private method in a Python class that converts all string dates in a specified collection of a MongoDB database
             to the datetime format.

        The method takes the name of the collection as an argument, and if verbose is True,
             it prints out a message indicating that the conversion is taking place.

        It then connects to the specified MongoDB database and collection, finds all documents in the collection,
            and loops through each document.
            For each document, it checks if the date field exists (indicating that the document has already been converted), 
            and if not, it uses a specific date format string to parse the created_at field of the document and convert it to a datetime object.
        
        It then updates the document in the collection with the new date field.
        
        """

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

    def __cloud_of_words(self, collection_name, verbose=False):
        """
        This is a method within a class that generates word clouds for a given collection in a MongoDB database.
        
        It first checks if the word cloud has already been generated and stored in the metadata collection of the database by comparing the count of tweets in different languages with the count stored in the metadata.
        
        If the count matches, it retrieves the stored word clouds from the metadata and displays them.
        
        Otherwise, it generates new word clouds for each language (French, English, and Arabic) by calling the __Wordcloud_language_generator method.
        
        After generating the word clouds, it updates the metadata with the new counts and word clouds.
        
        Finally, it displays the word clouds for each language in a subplot of a single figure.
        
        
        """
        
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
            wordcloud_fr, word_fr,freq_dist_fr = self.__Wordcloud_language_generator(lang="fr", collection_name=collection,
                                                                    verbose=verbose)
            print("English CW")
            wordcloud_en, word_en ,freq_dist_en= self.__Wordcloud_language_generator(lang="en", collection_name=collection,
                                                                    verbose=verbose)

            print("Arabic CW")
            wordcloud_ar, word_ar,freq_dist_ar = self.__Wordcloud_language_generator(lang="ar", collection_name=collection,
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

    def __tweets_lang_repartition(self, collection_name, verbose):
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

    def __plot_tweets_per_day(self, collection_name, verbose=False):
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

    def __extract_features(self, tokens, word_features, verbose=False):
        token_set = set(tokens)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in token_set)
        return features
    
    def __Train_arabic_classifier(self):
            """
            This code block is defining a method to train a Naive Bayes classifier for Arabic text classification.

            The method starts by reading in a CSV file containing text and their corresponding class labels (stance) and preprocessing 
                the text by removing stop words and stemming using the SnowballStemmer from the NLTK library.

            Next, it extracts features from the preprocessed text using a feature extraction method called __extract_features.
                This method takes in a list of tokens (words) and a list of word features and returns a dictionary of features for 
                the given text, where each feature is a boolean indicating whether the text contains the corresponding word feature.

            The code then creates a list of feature sets for the preprocessed data, where each feature set is a tuple containing 
                the extracted features and the corresponding class label. The data is split into a training and testing set using 
                the train_test_split function from the sklearn library.

            The Naive Bayes classifier is trained on the training set using the train method from the NLTK library.
             The accuracy of the classifier is then evaluated using the test set and printed out.
              
                Finally, the trained classifier is saved to a pickle file for later use and returned by the method.
            """
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

            featuresets = [(self.__extract_features(text, word_features), stance)
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

    def __Train_French_classifier(self):
            """
            This is a Python function that trains a French sentiment classifier using the Naive Bayes algorithm.
              The function reads a CSV file containing labeled French tweets,
                 preprocesses the data by removing stop words and stemming the remaining words,
                 extracts the 1000 most common words as features, splits the data into training and testing sets,
                 trains the classifier using the training set,
                 and saves the trained classifier to a file.
            """
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

            featuresets = [(self.__extract_features(text, word_features), stance)
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


    def __Train_English_classifier(self):
            """
            This is a Python function that trains an English sentiment classifier using the VADER (Valence Aware Dictionary and sEntiment Reasoner) algorithm.
              The function downloads the pre-trained VADER sentiment analyzer from the NLTK library,
                initializes it, and saves the trained analyzer to a file.

            The function saves the trained analyzer to a file called 'EnglishClassifier.pkl' using the pickle.dump method.
            The function returns the trained analyzer.
            """
          # preparing Classifier
            print("preparing  English Classifier", "\n")
            # Download the pre-trained sentiment analyzer
            nltk.download('vader_lexicon')

            # Initialize the sentiment analyzer
            sid = SentimentIntensityAnalyzer()
            with open('Transformer/Classifier/EnglishClassifier.pkl', 'wb') as f:
                pickle.dump(sid, f)
            return sid

    def __arabic_stance_classification(self,collection):
        """
         The method performs Arabic stance classification on the text data in the collection object.

        The method first checks if there is a preprocessed data file available,
            if yes, it loads the preprocessed data.
            If not, it reads data from a CSV file, preprocesses the data using stemming and tokenization,
              and saves the preprocessed data to a pickle file.

        The method then iterates through all the documents in the collection,
          extracts the text from each document,
            preprocesses it using stemming and tokenization,
              and extracts the features of the preprocessed text.
        It then classifies the text using a trained classifier and updates the "stance" field in the document with the predicted stance.

        Finally, the method counts the number of positive, negative, and neutral stances predicted and returns these counts.
        """
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
                features = self.__extract_features(filtered_tokens, word_features)

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
    
    def __french_stance_classification(self,collection):
        """
        This function takes a MongoDB collection as input and performs French stance classification on the text documents stored 
        in the collection.

        first it loads preprocessed data from a CSV file or preprocesses it if the preprocessed data is not available and stores
             it in a pickle file.
        Then, it extracts features from the preprocessed data using NLTK's FreqDist and word_features to get the 1000 most frequent words.

        Next, the function retrieves all French language documents from the MongoDB collection and performs 
        French stance classification on them.
            If the document already has a stance field, the function uses it,
            otherwise, it tokenizes the text, removes stop words and stems the remaining words before extracting features 
                        and classifying the stance using a trained classifier.

        Finally, the function updates the MongoDB collection by adding the stance field and returns the count of French documents
          that were classified as positive, negative, and neutral.
        
        """
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
                features = self.__extract_features(
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
    
    def __english_stance_classification(self,collection):
        """
        The  method is responsible for classifying the stances of English tweets using the VADER sentiment analysis tool.
        It retrieves all English tweets from the MongoDB collection,
        then for each tweet, it checks if the stance field exists.
            If it does, it uses the existing stance value,
            otherwise, it classifies the tweet's sentiment using VADER and updates the stance field with the predicted stance value.
             
        Finally, it returns the count of positive, negative, and neutral stances.
        """
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
    
    def __stance_language_repartition(self, collection_name, verbose=False):
        """
        This is a method in Python that handles the language repartition of documents within a collection.
         
        The function takes the name of a collection and a verbose flag as input parameters.
        It selects the database and collection to work with, then finds the metadata of the collection using a specific query.
        It also counts the number of documents in the collection with Arabic, French, and English languages.

        If the Arabic, French, and English document counts are different from their corresponding counts in the metadata,
            the method uses the __arabic_stance_classification(), __french_stance_classification(), and __english_stance_classification()
                functions to classify the documents' stance.
            It then updates the metadata with the new document counts and the corresponding stance classification counts.

        Finally, the method plots a bar chart for each language,
          indicating the positive, negative, and neutral stances in the corresponding language.
            If any of the languages have zero documents, no plot is produced for that language.
        
        
        """
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
                ar_positif, ar_negatif, ar_neutre=self.__arabic_stance_classification(collection=collection)

                
            if fr_count > 0:
                number_graphs = number_graphs + 1
                
                fr_positif, fr_negatif,  fr_neutre=self.__french_stance_classification(collection=collection)
            if en_count > 0:
                number_graphs = number_graphs + 1
                en_positif,en_negatif ,en_neutre =self.__english_stance_classification(collection=collection)
            
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


    
    
    def __localisation_distribution(self, collection_name, verbose=False):
        """
        The method takes a collection name as input and generates a bar chart showing the distribution of documents based on
        their location.
        
        It uses MongoDB's aggregation pipeline to extract the location data from documents and group them by location.
        
        It then sorts the results by the number of documents per location and displays the top 20 locations in the bar chart.
        
        The method also updates the metadata document in the collection with the distribution data to avoid running the pipeline 
        multiple times.
         
        The verbose argument is used to control the output messages during the method execution, and the method returns nothing.
        
        """


        # Select the database and collection
        db = self.DocGB_Driver.myclient[self.db_name]

        

        collection = db[str(collection_name)]
        
        count=collection.count_documents({})-1
        meta=collection.find_one({'_id': 'metadata'})
        if (meta['number_of_document']!= count or meta['doc_localisation_dist']==0):
            if verbose:
                print(f'Starting localisation with collection {collection_name}')

            pipeline = [
                
        {
            '$lookup': {
                'from': 'AlgeriaTwitterGraph', 
                'localField': 'user', 
                'foreignField': 'id_str', 
                'as': 'algeria_docs'
            }
        }, {
            '$unwind': {
                'path': '$algeria_docs', 
                'preserveNullAndEmptyArrays': True
            }
        }, {
            '$lookup': {
                'from': 'International_users', 
                'localField': 'user', 
                'foreignField': 'id_str', 
                'as': 'international_docs'
            }
        }, {
            '$unwind': {
                'path': '$international_docs', 
                'preserveNullAndEmptyArrays': True
            }
        }, {
            '$project': {
                'user_location': '$location', 
                'algeria_location': '$algeria_docs.location', 
                'international_location': '$international_docs.location'
            }
        }, {
            '$group': {
                '_id': {'$ifNull': ['$user_location', {'$ifNull': ['$algeria_location', '$international_location']}]},
                'count': {'$sum': 1}
            }
        }, {
            '$sort': {'count': -1}
        }, {
            '$limit': 20
        }
    ]

            cursor1 = collection.aggregate(pipeline)
            cursor=cursor1
            counts = {doc['_id']: doc['count'] for doc in tqdm(cursor,desc="getting top 20 localisations")}

            print(counts)
            

            # Step 7: Trier les emplacements par nombre de documents correspondants
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

            # Step 8: Conserver les 20 premiers emplacements les plus apparus
            top_locations = dict(sorted_counts[:20])

            cleaned_location_dict = {}

            for key, value in top_locations.items():
                    if(key):
                        print(key)
                        new_key = emoji.demojize(key)
                        cleaned_location_dict[new_key] = value

            for k, v in top_locations.items():
                if(k):
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
    
    
    def __create_metadoc(self, collection_name, verbose=False):
        """
        This code creates a metadata document for a MongoDB collection.

        It first selects the database and collection, then counts the number of documents in the collection and the number of documents
        with the "_id" field set to "metadata". It then initializes a dictionary called "document" with several fields and values,
        including the total number of documents in the collection, language and stance distributions, and date and location information.

        If the metadata document with "_id" set to "metadata" does not exist in the collection, the document is inserted.
        
            If the metadata document already exists in the collection, the fields in the existing document are updated with any 
            new values in the "document" dictionary using the update_one() method.

            If a new document has been added or deleted, the total number of documents field is also updated.
        """
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
        arabic_tweets_count=0
        topics=0

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
                    'doc_localisation_dist':doc_localisation_dist ,
                    'arabic_tweets_count':arabic_tweets_count,'arabic_topics':topics}
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

    def __Topic_detection(self,collection_name,verbose=False):

        db = self.DocGB_Driver.myclient[self.db_name]
        tweets_col = db[str(collection_name)]
        # Retrieve tweets from the MongoDB collection
        tweets_cursor = tweets_col.find({"lang": 'ar'})
        if verbose:
            print("number of arabic tweets : ",tweets_col.count_documents({"lang": 'ar'}))

        meta=tweets_col.find_one({'_id': 'metadata'})
        if (meta['arabic_tweets_count']!= tweets_col.count_documents({"lang": 'ar'}) or meta['topics']==0):
                
            # Create a list to store the tweet texts
            tweet_data = []

            # Iterate through the tweets and extract the text
            for tweet in tweets_cursor:
                if 'text' in tweet:
                    text = tweet['text']
                elif 'full_text' in tweet:
                    text = tweet['full_text']
                id_str = tweet['id_str']
                tweet_data.append({'id_str': id_str, 'text': text})

            # Create a pandas dataframe with the tweet data
            df = pd.DataFrame(tweet_data, columns=['id_str', 'text'])
            if verbose:
                print('Text Preprocessing  loading...')
            preprocessor = textPreprocessing()
            if verbose:
                print('Topic detection  loading...')
            topicDetectore= topicDetectionArabic()
            if verbose:
                print('Text Preprocessing ... ')
            clean_df=  preprocessor.preprocessing_arabic(df)

            if verbose:
                print('Topic predition  ... ')
                
            y_pred =  topicDetectore.PredictTopics(clean_df['text'])  
            df['y_pred']=y_pred
            # print(y_pred)
            for _, row in df.iterrows():
                
                # update the document with the new attribute
                tweets_col.update_one({'id_str': row['id_str']}, {'$set': {'topic': getTopic(row['y_pred'])}})
            
            y_pred_list=count_classes(y_pred)
            tweets_col.update_one({"_id": meta["_id"]}, {
                                        "$set": {'arabic_tweets_count':tweets_col.count_documents({"lang": 'ar'}),'arabic_topics':y_pred_list}})
            # print(y_pred_list)
        else:
            y_pred_list=meta['topics']
        plot_hist(y_pred_list)
        # plt.show()


    def pipeline(self,collection_name,remove_null=False,cloud_words=False,lang_dist=False,date_dist=False,stance_dist=False,localisation_dist=False,Topic_detection=False):
        """
        This is a pipeline function that executes different data processing steps on a MongoDB collection.
        
        The function takes a collection name and several boolean arguments as inputs, each representing a specific step in the pipeline.

        The steps include:

            1- Creating or updating a metadata document in the collection.
            2- Removing null values from the documents in the collection if the remove_null argument is True.
            3- Generating a word cloud if the cloud_words argument is True.
            4- Generating a distribution of tweets per language if the lang_dist argument is True.
            5- Generating a distribution of tweets per date if the date_dist argument is True.
            6- Generating a distribution of stance (positive, negative, neutral) per language if the stance_dist argument is True.
            7- Generating a distribution of tweets per location if the localisation_dist argument is True.
            8- Generating a topic detection  if the Topic_detection argument is True.
            
        At the end of the pipeline, the function displays any generated figures.
        
        """
        
        # create or update meta data document
        self.__create_metadoc(collection_name, verbose=True) 

        #remove null arguments from documents
        if(remove_null==True):
             self.__remove_and_update_null(collection_name,verbose=True)
        

        #number of tweets per localisation

        if(localisation_dist==True):
            self.__localisation_distribution(collection_name,verbose=True)

        
        #WordCloud generator
        if cloud_words==True:
            #update documents with adding tokens
            self.__doc_update_tokens(collection_name,verbose=True)
            #generation cloud of words
            self.__cloud_of_words(collection_name,verbose=True)

        #number of tweets per language
        if(lang_dist==True):
            self.__tweets_lang_repartition(collection_name,verbose=True)
        #number of tweets per date

        if(date_dist==True):
            self.__string_to_datetime(collection_name,verbose=True)
            self.__plot_tweets_per_day(collection_name,verbose=True)

       
        #stance repartition

        if(stance_dist==True):
            self.__stance_language_repartition(collection_name,verbose=True)

        
        #topic_detection

        if(Topic_detection==True):
            self.__Topic_detection(collection_name,verbose=True)

        #show figures
        plt.show()

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
    


    def meta___tweets_lang_repartition(self, verbose=False):
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




    def __localisation_distribution1(self, collection_name, verbose=False):
        client = pymongo.MongoClient()
        db = client[self.db_name]
        collection = db[collection_name]







        pipeline = [
            
    {
        '$lookup': {
            'from': 'AlgeriaTwitterGraph', 
            'localField': 'user', 
            'foreignField': 'id_str', 
            'as': 'algeria_docs'
        }
    }, {
        '$unwind': {
            'path': '$algeria_docs', 
            'preserveNullAndEmptyArrays': True
        }
    }, {
        '$lookup': {
            'from': 'International_users', 
            'localField': 'user', 
            'foreignField': 'id_str', 
            'as': 'international_docs'
        }
    }, {
        '$unwind': {
            'path': '$international_docs', 
            'preserveNullAndEmptyArrays': True
        }
    }, {
        '$project': {
            'user_location': '$location', 
            'algeria_location': '$algeria_docs.location', 
            'international_location': '$international_docs.location'
        }
    }, {
        '$group': {
            '_id': {'$ifNull': ['$user_location', {'$ifNull': ['$algeria_location', '$international_location']}]},
            'count': {'$sum': 1}
        }
    }, {
        '$sort': {'count': -1}
    }, {
        '$limit': 20
    }
]

        cursor1 = collection.aggregate(pipeline)
        cursor=cursor1
        counts = {doc['_id']: doc['count'] for doc in cursor}

        print(counts)
        

        # Step 7: Trier les emplacements par nombre de documents correspondants
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Step 8: Conserver les 20 premiers emplacements les plus apparus
        top_locations = dict(sorted_counts[:20])

        cleaned_location_dict = {}

        for key, value in top_locations.items():
                if(key):
                    print(key)
                    new_key = emoji.demojize(key)
                    cleaned_location_dict[new_key] = value

        for k, v in top_locations.items():
            if(k):
                k = k.strip()  # remove leading/trailing white space
                k = " ".join(k.split())  # replace multiple white space with single space
                cleaned_location_dict[k] = v
        
        top_locations=cleaned_location_dict   
        
        plt.style.use('ggplot')
        plt.barh(range(len(top_locations)), list(top_locations.values()), align='center')
        plt.yticks(range(len(top_locations)), list(top_locations.keys()))

        # Set the plot title and axis labels
        plt.title('Nombre de documents par emplacement')
        plt.xlabel('Nombre de documents')
        plt.ylabel('Emplacement')
        
        # Show the plot
        # plt.show()
    
        
