import string
from nltk.stem import SnowballStemmer
from pymongo import MongoClient
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from DataExtraction.DBHandlers import DocDBHandler



# Connect to MongoDB


class CloudOfWord():
    def __init__(self,connectionStringDocDB,mongo_db,mongo_collection,lang='english'):
            
            self.DocGB_Driver = DocDBHandler(connectionStringDocDB)
            self.db = self.DocGB_Driver.myclient[mongo_db]
            self.collection = self.db[mongo_collection]
            self.lang=lang
            self.stop_words = set(stopwords.words(lang))
            self.stemmer = SnowballStemmer(lang)
    def print_Could(self):

        # Extract tweets from MongoDB
        tweets = []
        for tweet in self.collection.find():
            if tweet['lang']==self.lang[:2]:
                tweets.append(tweet['full_text'])
        for i in range(0,5):
            print(tweets[i])  
        print(len(tweets))

        # Preprocess the text of the tweets
        processed_tweets = []
        for tweet in tweets:
            # Replace any URLs in the tweet with the string 'URL'
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'RT', '', tweet)
            # Tokenize the tweet into individual words
            words = nltk.word_tokenize(tweet.lower())
            # Remove stop words
            words = [w for w in words if w not in self.stop_words and w not in string.punctuation]
            # Stem or lemmatize the words
            words = [self.stemmer.stem(w) for w in words]
            # Join the words back into a single string
            processed_tweet = ' '.join(words)
            processed_tweets.append(processed_tweet)

        # Count the frequency of each word
        word_counts = {}
        for tweet in processed_tweets:
            for word in tweet.split():
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        # Filter out any words you don't want to include in the word cloud
        filtered_words = [word for word in word_counts if word not in self.stop_words and word_counts[word] > 5]

        # Create a word cloud from the most frequent words
        wordcloud = WordCloud(width=1600, height=800,font_path='font/kawkab-light.ttf', background_color='white').generate_from_frequencies(word_counts)

        # Display the word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()