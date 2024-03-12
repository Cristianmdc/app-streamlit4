import streamlit as st
import pickle
# Import necessary functions from your utility script or define them directly in this script
from model_utils import preprocess_and_predict  

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
    # Predict the class of user input
    result = preprocess_and_predict(user_input, model)  # Adjust according to your function's definition
    if result == 1:
        st.write("The text is classified as: Spam")
    else:
        st.write("The text is classified as: Not Spam")
