import streamlit as st
import pickle
  

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


