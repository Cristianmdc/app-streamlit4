import streamlit as st
import pickle
#load the model 
import pickle

with open('nlp.pkl','rb') as file:
  pickle.load(model, file)
    
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
