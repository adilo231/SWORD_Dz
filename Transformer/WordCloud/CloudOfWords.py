from wordcloud import WordCloud
from collections import Counter
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt

class CloudOfWords:
    def __init__(self):
        self.wordcloud=None
    def generateCW(self, list_of_tokens, language,path_arabic_font=None):
        if path_arabic_font != None :
           self.path_arabic_font=path_arabic_font
        if language == 'arabic':
            list_of_tokens_reshaped = [get_display(arabic_reshaper.reshape(token)) for token in list_of_tokens]
        else:
            list_of_tokens_reshaped = list_of_tokens


        word_freq = Counter(list_of_tokens_reshaped)

        if language == 'arabic':
            self.wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  font_path=self.path_arabic_font).generate_from_frequencies(word_freq)
        else:
            self.wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        return self.wordcloud

    def showCW(self):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
