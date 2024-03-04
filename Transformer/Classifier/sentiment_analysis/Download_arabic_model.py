import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("nourmorsy/PermoBERT-Arabic-Sentiment-Analysis-Farasa-BPE-44000Token")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("nourmorsy/PermoBERT-Arabic-Sentiment-Analysis-Farasa-BPE-44000Token")

# Define the directory where you want to save the model
save_directory = r"./Models/Arabic"

# Save the tokenizer and model
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
