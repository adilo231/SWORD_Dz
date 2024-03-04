from Sentiment_analysis import Sentiment_Analysis
sentiment_analyzer = Sentiment_Analysis()

arabic_text = "أنا سعيد جدًا اليوم!"
language = "ar"
sentiment = sentiment_analyzer.detect_sentiment(arabic_text, language)
print("Sentiment:", sentiment)

text = "I am feeling happy today!"
language = "en"
sentiment = sentiment_analyzer.detect_sentiment(text, language)
print("Sentiment:", sentiment)
