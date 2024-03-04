import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Sentiment_Analysis:
    def __init__(self):
       self.ar_sentiment_mapping = {
              0: 'negative',
              1: 'neutral',
              2: 'positive'
          }
       self.fr_en_sentiment_mapping = {
              0: 'negative',
              1: 'negative',
              2: 'neutral',
              3: 'positive',
              4: 'positive',
          }
      
       self.ar_tokenizer = AutoTokenizer.from_pretrained("./Models/Arabic")
       self.ar_model = AutoModelForSequenceClassification.from_pretrained("./Models/Arabic")
      
       self.fr_en_tokenizer = AutoTokenizer.from_pretrained("./Models/french-english")
       self.fr_en_model = AutoModelForSequenceClassification.from_pretrained("./Models/french-english")
        
    def detect_sentiment(self, text, language):
        if language == 'en' or language == 'fr':
            encoded_input = self.fr_en_tokenizer(text, return_tensors='pt')
            output = self.fr_en_model(**encoded_input)
            predicted_label = output.logits.argmax().item()
            return fr_en_sentiment_mapping[predicted_label]
        elif language == 'ar':
            encoded_input = self.ar_tokenizer(text, return_tensors='pt')
            output = self.ar_model(**encoded_input)
            predicted_label = output.logits.argmax().item()
            return ar_sentiment_mapping[predicted_label]
        else:
            return "none"
