import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Define the directory where you want to save the model
save_directory = "Models/french-english"

# Save the tokenizer and model
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
