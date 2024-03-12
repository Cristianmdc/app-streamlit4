import streamlit as st
import pickle
from nlp.pkl import preprocess_and_predict  # Ensure to replace 'your_script_name' with the actual name of your script file

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
    # Preprocess and predict the class of user input
    result = preprocess_and_predict(user_input, model)  # Adjust this if your preprocess_and_predict function doesn't use the model as input
    if result == 1:
        st.write("The text is classified as: Spam")
    else:
        st.write("The text is classified as: Not Spam")
