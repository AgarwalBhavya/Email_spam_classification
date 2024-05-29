import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    filtered_text = []
    for word in text:
        if word.isalnum():
            filtered_text.append(word)

    filtered_text = [word for word in filtered_text
                     if word not in stopwords.words('english') and word not in string.punctuation]

    stemmed_text = [ps.stem(word) for word in filtered_text]

    return " ".join(stemmed_text)

def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. Please ensure they exist in the same directory.")
        return None, None
model, tfidf = load_model_and_vectorizer()

#tfidf = pickle.load(open('vectorizer.pkl','rb'))
#model = pickle.load(open('model.pkl','rb'))

st.title('Email Spam Classifier')
input_sms=st.text_area("Enter the message")

if st.button('Predict'):
    #if model is None or tfidf is None:
     #   st.warning("Model or vectorizer not loaded. Please check for errors.")
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')