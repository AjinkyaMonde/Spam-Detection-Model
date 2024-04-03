import pandas as pd
df = pd.read_csv('./spam.csv', encoding = 'ISO-8859-1')
# df.head()
# df.info()
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df.rename(columns={'v1': 'label', 'v2': 'text'},inplace=True)
# df.head()
label_mapping = {'ham': 0, 'spam': 1}
df['label'] = df['label'].map(label_mapping)
# df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.33, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
featurizer = CountVectorizer(max_features=2000)
X_train_features = featurizer.fit_transform(X_train)
X_test_features = featurizer.transform(X_test)
nb_model = MultinomialNB()
nb_model.fit(X_train_features, y_train)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_features, y_train)
nb_y_pred = nb_model.predict(X_test_features)



def check(sample_text):
    data = pd.DataFrame({'text': [sample_text]})
    text_features = featurizer.transform(data['text'])
    return nb_model.predict(text_features)[0]
    


import streamlit as st
def main():
    st.title("Spam Detector")
    text = st.text_input("Enter Mail/SMS")
    if st.button("Predict"):
        if(check(text) == 1):
            st.error("Spam")
        else:
            st.success("Not Spam")

if __name__ == "__main__":
    main()