import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenization
    words = nltk.word_tokenize(text.lower())
    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(lemmatized_words)

def preprocess_and_predict(input_text, model):
    # Preprocess the input text
    cleaned_text = clean_and_lemmatize(input_text)
    # Make a prediction
    prediction = model.predict([cleaned_text])
    # Return the predicted class (0 or 1)
    return prediction[0]

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('nlp.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Setting title
st.title("Spam/Ham Classification App")

# Getting user input
user_input = st.text_area("Enter the text you want to classify:", "Type Here")

if st.button("Classify"):
    result = preprocess_and_predict(user_input, model)
    if result == 1:
        st.write("The text is classified as: Spam")
    else:
        st.write("The text is classified as: Not Spam")
