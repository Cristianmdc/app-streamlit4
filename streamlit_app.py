import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.stem import WordNetLemmatizer
import zipfile
import os

# You might need to comment out the download commands after running the app once, to avoid redownloading each time
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    # Remove email addresses, URLs, punctuation, numbers, and stopwords
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(lemmatized_words)

# Function to preprocess and predict input text
def preprocess_and_predict(input_text, model):
    cleaned_text = clean_and_lemmatize(input_text)
    prediction = model.predict([cleaned_text])
    return prediction[0]

# Assuming the model and vectorizer are already trained and saved (you'd have to save and load your model and vectorizer)
# For demonstration, let's assume you have a simple function to load a model (you should replace it with actual loading code)
def load_model():
    # Placeholder for model loading code
    # Example: return joblib.load('model_filename.pkl')
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

model = load_model()

# Streamlit app starts here
st.title('Email Spam Classification')
input_text = st.text_area("Enter the email text to classify as spam or not spam:", "Type Here...")

if st.button('Classify'):
    # Preprocess and predict
    predicted_class = preprocess_and_predict(input_text, model)
    if predicted_class == 1:
        st.success('The email is classified as Spam.')
    else:
        st.success('The email is classified as Not Spam.')

# Note on running the app:
# Ensure you have streamlit installed and run the app using the command:
# streamlit run streamlit_app.py
