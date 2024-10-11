import streamlit as st
from predictor import predict, load_model

# Set up the page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="LawyerAI - Legal Text Classifier", page_icon="⚖️")

# Load the Legal-BERT model with `AutoModelForPreTraining`
model = load_model()

# Streamlit Interface
st.title("LawyerAI - Legal Document Classifier ⚖️")
st.write("Enter the legal text to analyze its relevance or classification.")

# User Input
legal_text = st.text_area(label="Legal Text Input:", help="Input legal text to classify.")

# Prediction and Display
if legal_text:
    prediction = predict(model, legal_text)
    st.subheader(f"Prediction: {'Relevant' if prediction > 50 else 'Not Relevant'} ({round(prediction, 2)}%)")
