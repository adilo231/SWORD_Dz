from wordcloud import WordCloud , STOPWORDS
from collections import Counter
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import re

class CloudOfWords:
    def __init__(self,database,collection):
        self.docdbh= DocDBHandler(database,collection)
        self.tweets=self.docdbh.GetDocuments()
        self.wordcloud=None
        self.tokens=None
    def generateCW(self,language,path_arabic_font=None,max_words=None):
        if max_words == None :
            self.max_words=200
        else : 
            self.max_words=max_words
        self.tokens = []
        for tweet in self.tweets :
            try:
                if tweet['lang']==language :
                    self.tokens.extend(tweet['tokens'])
            except:
                pass
        

        if language == 'ar':
            if path_arabic_font != None :
                self.path_arabic_font=path_arabic_font
            self.tokens=[token for token in self.tokens if re.match(r'^[\u0600-\u06FF\s]+$', token)]
            list_of_tokens_reshaped = [get_display(arabic_reshaper.reshape(token)) for token in self.tokens]
        else:
            #this line is to remove non textual chars
            self.tokens = [re.sub(r'[^\w\s]', '', token) for token in self.tokens]
            list_of_tokens_reshaped = self.tokens


        word_freq = Counter(list_of_tokens_reshaped)

        if language == 'ar':
            self.wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  font_path=self.path_arabic_font,max_words = self.max_words, stopwords = set(STOPWORDS)).generate_from_frequencies(word_freq)
        else:
            self.wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    def showCW(self):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

