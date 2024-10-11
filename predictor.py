from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import streamlit as st

# Load Legal-BERT model using `AutoModelForSequenceClassification` and cache it as a resource
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")
    return tokenizer, model

# Prediction function for the classification model
def predict(model, text):
    tokenizer, model = model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    relevance_score = probabilities.detach().numpy()[0][1]  # Adjust index based on your labels
    return relevance_score * 100  # Scale to percentage
