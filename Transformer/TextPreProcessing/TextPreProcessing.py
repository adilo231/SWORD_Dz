from abc import ABC, abstractmethod
from __future__ import unicode_literals
import emoji
import asyncio
import nltk
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
import re
import spacy
from __future__ import unicode_literals
from pyarabic.araby import strip_tatweel
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS as english_stopwords
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.isri import ISRIStemmer



class TextPreProcessing(ABC):
    @abstractmethod
    def get_urls(self, text,lang):
        pass

    @abstractmethod
    def get_mentions(self, text,lang):
        pass

    @abstractmethod
    def get_hashtags(self, text,lang):
        pass

    @abstractmethod
    def get_emails(self, text,lang):
        pass


    @abstractmethod
    def remove_urls(self, text,lang):
        pass

    @abstractmethod
    def remove_mentions(self, text,lang):
        pass

    @abstractmethod
    def remove_hashtags(self, text,lang):
        pass

    @abstractmethod
    def remove_emails(self, text,lang):
        pass

    @abstractmethod
    def replace_emojis(self, text,lang):
        pass

    @abstractmethod
    def remove_punctuations(self, text,lang):
        pass

    @abstractmethod
    def remove_repeated_characters(self, text,lang):
        pass

    @abstractmethod
    def remove_multiple_spaces(self, text,lang):
        pass

    @abstractmethod
    def remove_non_usable_chars(self, text,lang):
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def stem_and_lemmatize(self, tokens):
        pass

    @abstractmethod
    def clean_text(self, text):
        pass


class EnglishTextProcessor(TextPreProcessing):
    def __init__(self):
        self.english_stemmer = PorterStemmer()
        self.nlp = spacy.load("en_core_web_sm")

    def get_hashtags(self, text):
        def split_hashtag_to_words(tag):
            tag = tag.replace('#', '')
            tags = tag.split('_')
            if len(tags) > 1:
                return tags
            pattern = re.compile(r"[A-Za-z]+|\d+")
            return pattern.findall(tag)

        def extract_hashtag(text):
            hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
            word_list = []
            for word in hash_list:
                word_list.extend(split_hashtag_to_words(word))
            return word_list
        
        return extract_hashtag(text)

    def get_emails(self,text):
        emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', text)

        return emails

    def get_mentions(self, text):
        mentions = re.findall(r'@(\w+)', text)
        return mentions

    def get_urls(self, text):
        return re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',text)

    def remove_hashtags(self, text):
        def clean_hashtag(text):
            words = text.split()
            cleaned_text = []
            for word in words:
                if not is_hashtag(word):
                    cleaned_text.append(word)
            return " ".join(cleaned_text)

        def is_hashtag(word):
            return word.startswith("#")

        return clean_hashtag(text)

    def remove_emails(self, text):
        return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)',"",text)

    def remove_mentions(self, text):
        mention_pattern = r'@+'
        cleaned_text = re.sub(mention_pattern, '', text)
        return cleaned_text

    def replace_emojis(self, text):
        modified_text = emoji.demojize(text)
        return modified_text

    def remove_urls(self, text):
        return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',"",text)

    def remove_punctuations(self, text):
        return re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)

    def remove_stopwords(self, text):
        return " ".join([t for t in text.split() if t.lower() not in english_stopwords])

    def remove_non_usable_chars(self, text):
        english_pattern = re.compile(r'[^\x00-\x7F\d\s]+')    
        cleaned_text = english_pattern.sub('', text)
        return cleaned_text

    def remove_repeated_characters(self, text):
        repeated_pattern = re.compile(r'(\S)(\1)+', re.UNICODE)
        cleaned_text = repeated_pattern.sub(r'\1', text)
        return cleaned_text

    def remove_multiple_spaces(self, text):
        return ' '.join(text.split())

    def tokenize(self, text):
        doc = self.nlp(text)
        return doc

    def stem_and_lemmatize(self, tokens):
        return [self.english_stemmer.stem(token.lemma_) if token.lemma_ != "-PRON-" and token.lemma_ != "be" else token.text for token in tokens]   

    def correct_text(self,text):
        return str(TextBlob(text).correct())

    def clean_text(self, text):
        text=self.correct_text(text)
        text=self.remove_emails(text)
        text=self.remove_hashtags(text)
        text=self.remove_mentions(text)
        text=self.remove_urls(text)
        text=self.replace_emojis(text)
        text=self.remove_punctuations(text)
        text=self.remove_repeated_characters(text)
        text=self.remove_multiple_spaces(text)
        text=self.remove_stopwords(text)
        tokens=self.tokenize(text)
        return self.stem_and_lemmatize(tokens)


class FrenchTextProcessor(TextPreProcessing):
    def __init__(self):
        with open('Selected_Fr_emojis.csv','r',encoding='utf-8') as f:
            lines = f.readlines()
            self.emojis_fr = {}
            for line in lines:
                line = line.strip('\n').split(';')
                self.emojis_fr.update({line[0].strip():line[1].strip()})
        self.french_stemmer = SnowballStemmer('french')
        self.french_stopwords = set(nltk.corpus.stopwords.words('french'))
        self.nlp = spacy.load("fr_core_news_sm")



    def get_hashtags(self, text):
        def split_hashtag_to_words(tag):
            tag = tag.replace('#', '')
            tags = tag.split('_')
            if len(tags) > 1:
                return tags
            pattern = re.compile(r"[A-Za-z]+|\d+")
            return pattern.findall(tag)

        def extract_hashtag(text):
            hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
            word_list = []
            for word in hash_list:
                word_list.extend(split_hashtag_to_words(word))
            return word_list
        
        return extract_hashtag(text)

    def get_emails(self,text):
        emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', text)

        return emails

    def get_mentions(self, text):
        mentions = re.findall(r'@(\w+)', text)
        return mentions

    def get_urls(self, text):
        return re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',text)

    def remove_hashtags(self, text):
        def clean_hashtag(text):
            words = text.split()
            cleaned_text = []
            for word in words:
                if not is_hashtag(word):
                    cleaned_text.append(word)
            return " ".join(cleaned_text)

        def is_hashtag(word):
            return word.startswith("#")

        return clean_hashtag(text)

    def remove_emails(self, text):
        return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)',"",text)

    def remove_mentions(self, text):
        mention_pattern = r'@+'
        cleaned_text = re.sub(mention_pattern, '', text)
        return cleaned_text

    def replace_emojis(self, text):
        def remove_emoji(text):
            emoji_pattern = re.compile("["
                                        u"\U0001F600-\U0001F64F"  
                                        u"\U0001F300-\U0001F5FF" 
                                        u"\U0001F680-\U0001F6FF" 
                                        u"\U0001F1E0-\U0001F1FF" 
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
            return text
                
        def emoji_native_translation(text):
            text = text.lower()
            loves = ["<3", "♥",'❤']
            smilefaces = []
            sadfaces = []
            neutralfaces = []

            eyes = ["8",":","=",";"]
            nose = ["'","`","-",r"\\"]
            for e in eyes:
                for n in nose:
                    for s in ["\)", "d", "]", "}","p"]:
                        smilefaces.append(e+n+s)
                        smilefaces.append(e+s)
                    for s in ["\(", "\[", "{"]:
                        sadfaces.append(e+n+s)
                        sadfaces.append(e+s)
                    for s in ["\|", "\/", r"\\"]:
                        neutralfaces.append(e+n+s)
                        neutralfaces.append(e+s)
                    #reversed
                    for s in ["\(", "\[", "{"]:
                        smilefaces.append(s+n+e)
                        smilefaces.append(s+e)
                    for s in ["\)", "\]", "}"]:
                        sadfaces.append(s+n+e)
                        sadfaces.append(s+e)
                    for s in ["\|", "\/", r"\\"]:
                        neutralfaces.append(s+n+e)
                        neutralfaces.append(s+e)

            smilefaces = list(set(smilefaces))
            sadfaces = list(set(sadfaces))
            neutralfaces = list(set(neutralfaces))
            t = []
            for w in text.split():
                if w in loves:
                    t.append("Amour")
                elif w in smilefaces:
                    t.append("Sourire")
                elif w in neutralfaces:
                    t.append("Neutre")
                elif w in sadfaces:
                    t.append("Triste")
                else:
                    t.append(w)
            newText = " ".join(t)
            return newText


        def is_emoji(word):
            if word in self.emojis_fr:
                return True
            else:
                return False
            
        def add_space(text):
            return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()
        
        loop = asyncio.get_event_loop()

        def translate_emojis(words):
            word_list = list()
            words_to_translate = list()
            for word in words :
                t = self.emojis_fr.get(word.get('emoji'),None)
                if t is None:
                    word.update({'translation':'normale','translated':True})
                    #words_to_translate.append('normal')
                else:
                    word.update({'translated':False,'translation':t})
                    words_to_translate.append(t.replace(':','').replace('_',' '))
                word_list.append(word)
            return word_list

        def emoji_unicode_translation(text):
            text = add_space(text)
            words = text.split()
            text_list = list()
            emojis_list = list()
            c = 0
            for word in words:
                if is_emoji(word):
                    emojis_list.append({'emoji':word,'emplacement':c})
                else:
                    text_list.append(word)
                c+=1
            emojis_translated = translate_emojis(emojis_list)
            for em in emojis_translated:
                text_list.insert(em.get('emplacement'),em.get('translation'))
            text = " ".join(text_list)
            return text

        text = emoji_native_translation(text)
        text = emoji_unicode_translation(text)
        return text
    
    def remove_urls(self, text):
        return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',"",text)

    def remove_punctuations(self, text):
        return re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)

    def remove_stopwords(self, text):
        return " ".join([t for t in text.split() if t.lower() not in self.french_stopwords])

    def remove_non_usable_chars(self, text):
        french_pattern = re.compile(r'[^\s\dA-Za-zàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]+')
        cleaned_text = french_pattern.sub('', text)
        return cleaned_text

    def remove_repeated_characters(self, text):
        repeated_pattern = re.compile(r'(\S)(\1)+', re.UNICODE)
        cleaned_text = repeated_pattern.sub(r'\1', text)
        return cleaned_text

    def remove_multiple_spaces(self, text):
        return ' '.join(text.split())

    def tokenize(self, text):
        doc = self.nlp(text)
        return doc

    def stem_and_lemmatize(self, tokens):
        tokens_lemmas = [token.lemma_ if token.lemma_ != "-PRON-" and token.lemma_ != "be" else token.text for token in tokens]
        return tokens_lemmas

    def clean_text(self, text):
        text=self.remove_emails(text)
        text=self.remove_hashtags(text)
        text=self.remove_mentions(text)
        text=self.remove_urls(text)
        text=self.replace_emojis(text)
        text=self.remove_punctuations(text)
        text=self.remove_repeated_characters(text)
        text=self.remove_multiple_spaces(text)
        text=self.remove_stopwords(text)
        tokens=self.tokenize(text)
        text=self.stem_and_lemmatize(tokens)
        return text


class ArabicTextProcessor(TextPreProcessing):

    def __init__(self):
        with open('emojis.csv','r',encoding='utf-8') as f:
            lines = f.readlines()
            self.emojis_ar = {}
            for line in lines:
                line = line.strip('\n').split(';')
                self.emojis_ar.update({line[0].strip():line[1].strip()})


        self.arabic_stemmer = ISRIStemmer()
        self.arabic_stopwords = stopwords.words('arabic')
        self.nlp = spacy.load("fr_core_news_sm")

    def get_urls(self, text):
        urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',text)
        return urls

    def get_mentions(self, text):
        mentions = re.findall(r'@(\w+)', text)
        return mentions

    def get_hashtags(self, text):
        def split_hashtag_to_words(tag):
            tag = tag.replace('#', '')
            tags = tag.split('_')
            if len(tags) > 1:
                return tags
            else:
                return [tag]
            
        hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
        word_list = []
        for word in hash_list:
            word_list.extend(split_hashtag_to_words(word))
        return word_list

    def get_emails(self,text):
        emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', text)
        return emails

    def remove_urls(self, text):
        return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',"",text)

    def remove_mentions(self, text):
        mention_pattern = r'@+'
        cleaned_text = re.sub(mention_pattern, '', text)
        return cleaned_text

    def remove_hashtags(self, text):
        def is_hashtag(word):
            if word.startswith("#"):
                return True
            else:
                return False
            
        words = text.split()
        text = list()
        for word in words:
            if is_hashtag(word):
                pass
            else:
                text.append(word)
        return " ".join(text)

    def remove_emails(self, text):
        return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)',"",text)

    def remove_punctuations(self, text):
        return re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)

    def remove_repeated_characters(self, text):
        repeated_pattern = re.compile(r'(\S)(\1)+', re.UNICODE)
        cleaned_text = repeated_pattern.sub(r'\1', text)
        return cleaned_text

    def remove_multiple_spaces(self, text):
        return ' '.join(text.split())

    def remove_non_usable_chars(self, text):
        arabic_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\d\s]+')
        cleaned_text = arabic_pattern.sub('', text)
        return cleaned_text

    def normalizeArabic(self,text):
        text = text.strip()
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        noise = re.compile(""" ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
        text = re.sub(noise, '', text)
        text = re.sub(r'(.)\1+', r"\1\1", text) 
        return text

    def tokenize(self, text):
        doc = self.nlp(text)
        return [token.text for token in doc]

    def stem_and_lemmatize(self, word):
        return self.arabic_stemmer.stem(word)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.arabic_stopwords]

    def replace_emojis(self, text):
        def remove_emoji(text):
            emoji_pattern = re.compile("["
                                        u"\U0001F600-\U0001F64F"  
                                        u"\U0001F300-\U0001F5FF" 
                                        u"\U0001F680-\U0001F6FF" 
                                        u"\U0001F1E0-\U0001F1FF" 
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
            return text
                
        def emoji_native_translation(text):
            text = text.lower()
            loves = ["<3", "♥",'❤']
            smilefaces = []
            sadfaces = []
            neutralfaces = []

            eyes = ["8",":","=",";"]
            nose = ["'","`","-",r"\\"]
            for e in eyes:
                for n in nose:
                    for s in ["\)", "d", "]", "}","p"]:
                        smilefaces.append(e+n+s)
                        smilefaces.append(e+s)
                    for s in ["\(", "\[", "{"]:
                        sadfaces.append(e+n+s)
                        sadfaces.append(e+s)
                    for s in ["\|", "\/", r"\\"]:
                        neutralfaces.append(e+n+s)
                        neutralfaces.append(e+s)
                    #reversed
                    for s in ["\(", "\[", "{"]:
                        smilefaces.append(s+n+e)
                        smilefaces.append(s+e)
                    for s in ["\)", "\]", "}"]:
                        sadfaces.append(s+n+e)
                        sadfaces.append(s+e)
                    for s in ["\|", "\/", r"\\"]:
                        neutralfaces.append(s+n+e)
                        neutralfaces.append(s+e)

            smilefaces = list(set(smilefaces))
            sadfaces = list(set(sadfaces))
            neutralfaces = list(set(neutralfaces))
            t = []
            for w in text.split():
                if w in loves:
                    t.append("حب")
                elif w in smilefaces:
                    t.append("مضحك")
                elif w in neutralfaces:
                    t.append("عادي")
                elif w in sadfaces:
                    t.append("محزن")
                else:
                    t.append(w)
            newText = " ".join(t)
            return newText


        def is_emoji(word):
            if word in emojis_ar:
                return True
            else:
                return False
            
        def add_space(text):
            return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()
        
        loop = asyncio.get_event_loop()

        def translate_emojis(words):
            word_list = list()
            words_to_translate = list()
            for word in words :
                t = emojis_ar.get(word.get('emoji'),None)
                if t is None:
                    word.update({'translation':'عادي','translated':True})
                    #words_to_translate.append('normal')
                else:
                    word.update({'translated':False,'translation':t})
                    words_to_translate.append(t.replace(':','').replace('_',' '))
                word_list.append(word)
            return word_list

        def emoji_unicode_translation(text):
            text = add_space(text)
            words = text.split()
            text_list = list()
            emojis_list = list()
            c = 0
            for word in words:
                if is_emoji(word):
                    emojis_list.append({'emoji':word,'emplacement':c})
                else:
                    text_list.append(word)
                c+=1
            emojis_translated = translate_emojis(emojis_list)
            for em in emojis_translated:
                text_list.insert(em.get('emplacement'),em.get('translation'))
            text = " ".join(text_list)
            return text

        text = emoji_native_translation(text)
        text = emoji_unicode_translation(text)
        return text

    def remove_diacritics(self, word):
        return strip_tashkeel(word)

    def remove_tatweel(self, word):
        return strip_tatweel(word)


    def clean_text(self,text):
        text = self.remove_hashtags(text)
        text = self.replace_emojis(text)
        text = self.remove_urls(text)
        text = self.remove_punctuations(text)
        text = self.remove_non_usable_chars(text)
        text = self.remove_repeated_characters(text)
        text = self.remove_multiple_spaces(text)
        tokens = self.tokenize(text)
        cleaned_tokens = self.remove_stopwords(tokens)
        for i, word in enumerate(cleaned_tokens):
            diacritic_removed_word = self.remove_diacritics(word)
            final_word = self.remove_tatweel(diacritic_removed_word)
            normalized_word = self.normalizeArabic(final_word)
            cleaned_tokens[i] = normalized_word
        return cleaned_tokens

