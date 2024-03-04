from TextPreProcessing import FrenchTextProcessor,EnglishTextProcessor,ArabicTextProcessor
# Create an instance of EnglishTextProcessor
english_processor = EnglishTextProcessor()

# Sample English text
sample_text = "This is an example text. It contains some #hashtags, @mentions, and emojis 😊. You can also find some URLs like http://example.com"

# Clean the text using the clean_text method
cleaned_tokens = english_processor.clean_text(sample_text)

# Print the cleaned tokens
print(cleaned_tokens)

# Create an instance of FrenchTextProcessor
french_processor = FrenchTextProcessor()

# Sample French text
sample_text = "Ceci est un exemple de texte français. Il contient quelques #hashtags et des @mentions. Vous pouvez également y trouver des émojis 😊 et des liens http://example.com"

# Clean the text using the clean_text method
cleaned_text = french_processor.clean_text(sample_text)

# Print the cleaned text
print(cleaned_text)

# Create an instance of ArabicTextProcessor
arabic_processor = ArabicTextProcessor()

# Sample Arabic text
sample_text = "هذا هو نص عينة #نص_عينة 😊 يحتوي على بعض الرموز التعبيرية والرابط https://example.com"

# Clean the text using the clean_text method
cleaned_text = arabic_processor.clean_text(sample_text)

# Print the cleaned text
print(cleaned_text)
