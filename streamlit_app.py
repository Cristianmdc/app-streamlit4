import streamlit as st
import pickle

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('nlp.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()
# Modify this function to accept the model as an argument
def preprocess_and_predict(input_text, model):
    cleaned_text = clean_and_lemmatize(input_text)
    prediction = model.predict([cleaned_text])
    return prediction[0]

# Setting title
st.title("Spam/Ham Classification App")

# Define or import your clean_and_lemmatize function here as well

# Getting user input
user_input = st.text_area("Enter the text you want to classify:", "Type Here")

if st.button("Classify"):
    # Ensure the preprocess_and_predict function is defined or imported correctly
    result = preprocess_and_predict(user_input, model)
    if result == 1:
        st.write("The text is classified as: Spam")
    else:
        st.write("The text is classified as: Not Spam")
