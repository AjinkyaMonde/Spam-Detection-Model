import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

pickle_in = open("vectorizer.pkl","wb")
vectorizer = pickle.load(pickle_in)

def load_model():
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    # Assuming you have saved your model as 'model.pkl'
    model.fit(df['text'], df['label'])
    return model

def predict_spam(input_text, model):
    prediction = model.predict([input_text])
    return prediction[0]

    # Load your dataset
    df = pd.read_csv("spam.csv")

# Load the model
model = load_model()

def main():

     st.title("Spam Detection Model")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    # st.markdown(html_temp,unsafe_allow_html=True)
    # variance = st.text_input("Variance","Type Here")
    # skewness = st.text_input("skewness","Type Here")
    # curtosis = st.text_input("curtosis","Type Here")
    # entropy = st.text_input("entropy","Type Here")
    # result=""
    if st.button("Spam"):
        result=predict_spam(input_text, model)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

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




if __name__=='__main__':
    main()