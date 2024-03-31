import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# Function to load the model
@st.cache
def load_model():
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    # Assuming you have saved your model as 'spam_detection_model.pkl'
    model.fit(df['text'], df['label'])
    return model

# Function to predict spam or ham
def predict_spam(input_text, model):
    prediction = model.predict([input_text])
    return prediction[0]

# Load your dataset
df = pd.read_csv("spam.csv")

# Load the model
model = load_model()

# Streamlit UI
st.title("Spam Detection App")

user_input = st.text_input("Enter a message:")
if st.button("Predict"):
    if user_input:
        prediction = predict_spam(user_input, model)
        if prediction == 'ham':
            st.success("This message is not spam.")
        else:
            st.error("This message is spam.")
    else:
        st.warning("Please enter a message.")

def load_model():
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    # Assuming you have saved your model as 'spam_detection_model.pkl'
    model.fit(df['text'], df['label'])
    return model

# Function to predict spam or ham
def predict_spam(input_text, model):
    prediction = model.predict([input_text])
    return prediction[0]

# Load your dataset
df = pd.read_csv("spam.csv")

# Load the model
model = load_model()

# Streamlit UI
st.title("Spam Detection App")

user_input = st.text_input("Enter a message:")
if st.button("Predict"):
    if user_input:
        prediction = predict_spam(user_input, model)
        if prediction == 'ham':
            st.success("This message is not spam.")
        else:
            st.error("This message is spam.")
    else:
        st.warning("Please enter a message.")
