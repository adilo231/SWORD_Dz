from wordcloud import WordCloud
from collections import Counter
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import re
from DataStorage.DocDBHandler import DocDBHandler 
class CloudOfWords:
    def __init__(self,database,collection):
        self.docdbh= DocDBHandler(database,collection)
        self.tweets=self.docdbh.GetDocuments()
        self.wordcloud=None
        self.tokens=None
    def generateCW(self,language,path_arabic_font=None):
        self.tokens = []
        for tweet in tweets :
            try:
                if tweet['lang']==language :
                    self.tokens.extend(tweet['tokens'])
            except:
                pass
        #this line is to remove non textual chars
        self.tokens = [re.sub(r'[^\w\s]', '', token) for token in self.tokens]
        if path_arabic_font != None :
           self.path_arabic_font=path_arabic_font
        if language == 'ar':
            list_of_tokens_reshaped = [get_display(arabic_reshaper.reshape(token)) for token in self.tokens]
        else:
            list_of_tokens_reshaped = self.tokens


        word_freq = Counter(list_of_tokens_reshaped)

        if language == 'ar':
            self.wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  font_path=self.path_arabic_font).generate_from_frequencies(word_freq)
        else:
            self.wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    def showCW(self):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

