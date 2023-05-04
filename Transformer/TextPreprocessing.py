import warnings
warnings.filterwarnings("ignore")
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
import pandas as pd


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.cm as cm
from matplotlib import rcParams




from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.isri import ISRIStemmer
from collections import Counter 
import itertools
import re
import string
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from joblib import  load
def getTopic(num):
    topics=['culture', 'diverse', 'economy', 'politic', 'sport']
    return topics[num]
def getTopicIndex(topic):
    topics=['culture', 'diverse', 'economy', 'politic', 'sport']
    return topics.index(topic)

plt.style.use('ggplot')

def count_classes(y_pred):
    # Define the class labels
    class_labels = ['culture', 'diverse', 'economy', 'politic', 'sport']
    
    # Initialize the count for each class to zero
    class_counts = {label: 0 for label in class_labels}
    
    # Count the number of predictions for each class
    for pred in y_pred:
        class_counts[class_labels[pred]] += 1
    
    return class_counts

def plot_hist(y_pred_counts):
    # Define the class labels
    class_labels = ['culture', 'diverse', 'economy', 'politic', 'sport']

    # Extract the count for each class
    counts = [y_pred_counts[label] for label in class_labels]

    # Plot the histogram
    plt.bar(range(len(class_labels)), counts)
    plt.xticks(range(len(class_labels)), class_labels)
    plt.title('Histogram of Class Predictions')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
def countPropetries(df):
    all_words = [word for tokens in df["text"] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in df["text"]]

    VOCAB = sorted(list(set(all_words)))

    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))
    return all_words
def plot(all_words, title):
    
    
    counted_words = Counter(all_words)

    words = []
    counts = []
    for letter, count in counted_words.most_common(25):
        words.append(letter)
        counts.append(count)

    colors = cm.rainbow(np.linspace(0, 1, 10))
    rcParams['figure.figsize'] = 20, 10
    
    

    # Reshape Arabic text
    reshaped_texts = [arabic_reshaper.reshape(word) for word in words]

    # Get displayable Arabic text
    displayable_text = [get_display(reshaped_text) for reshaped_text in reshaped_texts]
    
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Words')
    font = {'family': 'DejaVu Sans', 'size': 12}
    plt.barh(displayable_text, counts, color=colors)
    plt.show()

class topicDetectionArabic():
    def __init__(self):
        self.model = load("Data/Models/svm_model.joblib")
    def PredictTopics(self,texts):
         y_pred = self.model.predict(texts.astype('str'))
         return y_pred

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
class textPreprocessing():
    def __init__(self):

        self.punctuations_list = arabic_punctuations + english_punctuations

    def preprocessing_arabic(self,df):
        clean_df = df.dropna()
        stemmer = ISRIStemmer()
        clean_df["text"] = clean_df['text'].apply(lambda x: self._processDocument(x, stemmer))
        tokenizer = RegexpTokenizer(r'\w+')
        clean_df["text"] = clean_df["text"].apply(tokenizer.tokenize)
        stopwords_list = stopwords.words('arabic')
        
        print
        clean_df["text"] = clean_df["text"].apply(lambda x: [item for item in x if item not in stopwords_list])
        return clean_df




    def _remove_hashtag(self,df, col = 'text'):
        for letter in r'#.][!XR':
            df[col] = df[col].astype(str).str.replace(letter,'', regex=True)
        
    def _remove_punctuations(self,text):
        translator = str.maketrans('', '', self.punctuations_list)
        return text.translate(translator)  

    def _normalize_arabic(self,text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        return text     


    def _remove_repeating_char(self,text):
        return re.sub(r'(.)\1+', r'\1', text)


    def _processDocument(self,doc, stemmer): 

        #Replace @username with empty string
        doc = re.sub(r'@[^\s]+', ' ', doc)
        doc = re.sub(r'_', ' ', doc)
        doc = re.sub(r'\n', ' ', doc)
        doc = re.sub(r'[a-z,A-Z]', '', doc)
        doc = re.sub(r'\d', '', doc)
        #Convert www.* or https?://* to " "
        doc = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',doc)
        #Replace #word with word
        doc = re.sub(r'#([^\s]+)', r'\1', doc)
        # remove punctuations
        doc= self._remove_punctuations(doc)
        # normalize the tweet
        doc= self._normalize_arabic(doc)
        # remove repeated letters
        doc=self._remove_repeating_char(doc)
        #stemming
        doc = stemmer.stem(doc)
        
        return doc




from pymongo import MongoClient

if __name__ == '__main__':
    # df = pd.read_csv("Data/arabic_dataset_classifiction.csv")
    # clean_df=preprocessing_arabic(df.sample(100))
    # culture_df = clean_df.loc[clean_df["targe"] == 0]
    # diverse_df = clean_df.loc[clean_df["targe"] == 1]
    # economy_df = clean_df.loc[clean_df["targe"] == 2]
    # politic_df = clean_df.loc[clean_df["targe"] == 3]
    # sport_df = clean_df.loc[clean_df["targe"] == 4]
    # print(culture_df.shape,diverse_df.shape,economy_df.shape,politic_df.shape,sport_df.shape)
    # culture_words = countPropetries(culture_df)
    # print("Culture : ")
   
    # print("\nDiverse : ")
    # diverse_words = countPropetries(diverse_df)
    # print("\nEconomy : ")
    # economy_words = countPropetries(economy_df)
    # print("\nPolitics : ")
    # politic_words = countPropetries(politic_df)
    # print("\nSport : ")
    # sport_words = countPropetries(sport_df)

    # plot(culture_words, 'Top words in Culture')
    # print('Data loading... ')
    # with open('Data/cleanedData.pkl', 'rb') as f:
    #     clean_df = pickle.load(f)
    # clean_df=clean_df.sample(200)
    # y = clean_df['targe']
    # X = clean_df['text']
    # print(clean_df.head())
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    # print('model loading... ')
    
    # model = load("svm_model.joblib")
    # y_pred = model.predict(X_test.astype('str'))
    # print(y_pred)
    # result = calculate_results(y_test, y_pred)
    # print(result)
    # cm = confusion_matrix(y_test, y_pred)
    # # plot_confusion_matrix(cm, ['culture', 'diverse', 'economy', 'politic', 'sport'])
    # plot_hist(y_pred)
    

   


    # Connect to the MongoDB database
    client = MongoClient('mongodb://localhost:27017/')
    db = client['twitter_db']
    tweets_col = db['Prix']


    # Retrieve tweets from the MongoDB collection
    tweets_cursor = tweets_col.find({"lang": 'ar'})
    print(tweets_col.count_documents({"lang": 'ar'}))
    # Create a list to store the tweet texts
    tweet_data = []
    print('1- Starting')
    # Iterate through the tweets and extract the text
    for tweet in tweets_cursor:
        if 'text' in tweet:
            text = tweet['text']
        elif 'full_text' in tweet:
            text = tweet['full_text']
        id_str = tweet['id_str']
        tweet_data.append({'id_str': id_str, 'text': text})
    print('Starting')
    # Create a pandas dataframe with the tweet data
    df = pd.DataFrame(tweet_data, columns=['id_str', 'text'])
    print('Text Preprocessing  loading...')
    preprocessor = textPreprocessing()
    print('Topic detection  loading...')
    topicDetectore= topicDetectionArabic()

    print('Text Preprocessing ... ')
    clean_df=  preprocessor.preprocessing_arabic(df)
    

    print('Topic predition  ... ')
    y_pred =  topicDetectore.PredictTopics(clean_df['text'])  
    plot_hist(y_pred)

    df['y_pred']=y_pred
    print(df.head())
    for _, row in df.iterrows():
        print(row)
        # update the document with the new attribute
        tweets_col.update_one({'id_str': row['id_str']}, {'$set': {'topic': getTopic(row['y_pred'])}})
