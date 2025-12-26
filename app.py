import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")

#if tfidf true or false 
#tfidf = model.named_steps["tfidf"]
#st.write("TF-IDF fitted:", hasattr(tfidf, "idf_"))

label_map = {
    0: "Sad 😢",
    1: "Happy 😄",
    2: "Love 💖",
    3: "Angry 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}

st.title("Sentiment Analyzer")

text = st.text_area("Enter a sentence")

if st.button("Analyze"):
    if text.strip():
        pred = model.predict([text])[0]
        st.success(label_map[pred])
    else:
        st.warning("Please enter text")
