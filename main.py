import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import gdown
import os

# List the file IDs of the necessary files in your Google Drive folder
file_ids = {
    "config.json": "1R7cJg2_iPemfmVzTMqOQMEo7Omlvvzte",  # Replace with actual file ID for config.json
    "model.safetensors": "12AgOktT6CYIi6a6bAobcdVJrkwTBey0Z",  # Replace with actual file ID for model.safetensors
    "special_tokens_map.json": "1A91zZ8H3J_RNcx67mQQeSt0uVkUoeySC",  # Replace with actual file ID
    "tokenizer_config.json": "1OQe5Kv050_5KReoZYac3lwsq5N1CSXxW",  # Replace with actual file ID
    "vocab.txt": "1VmjF3i9qvPUEDZOLmQ9vYuc69vQoB1Ie"  # Replace with actual file ID for vocab.txt
}

model_dir = "distilbert_spam_model"

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Download each file using its file ID
for file_name, file_id in file_ids.items():
    file_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(file_url, os.path.join(model_dir, file_name), quiet=False)

# Load the model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

# Function to classify spam or not
def classify_spam(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit UI
st.set_page_config(page_title="Cybersecurity Spam Checker", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.write("Welcome to the Cybersecurity Spam Checker!")
st.sidebar.write("Use this tool to classify messages as spam or not.")

# Main title
st.title("Cybersecurity Spam Checker")
st.write("### Enter a message to classify whether it is spam or not.")

# Text input for the message
user_input = st.text_area("Message", height=150)

# Classify button
if st.button("Classify"):
    if user_input:
        result = classify_spam(user_input)
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter a message to classify.")

# Additional information section
st.markdown("---")
st.subheader("About This Tool")
st.write(""" 
This tool uses a DistilBERT model to classify messages as spam or not. 
Spam messages can be harmful and may contain phishing attempts or unwanted advertisements. 
By using this tool, you can help protect yourself from potential threats.
""")

st.subheader("How It Works")
st.write(""" 
1. Enter a message in the text area above. 
2. Click the "Classify" button. 
3. The model will analyze the message and provide a prediction. 
""")

st.subheader("Disclaimer")
st.write(""" 
This tool is for educational purposes only. 
While it aims to provide accurate classifications, 
Always use caution when dealing with unknown messages. 
""")


