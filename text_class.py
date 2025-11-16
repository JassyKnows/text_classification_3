import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle




def sentiment_model():
    sentiment_labels = ["Positive","Negative","Neutral","Irrelevant"]
    @st.cache_resource
    def load_sentiment_model():
        model = load_model("sentiment_model.keras")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    
    model, tokenizer = load_sentiment_model()
    user_text = st.text_area("Enter a sentence to analyze its sentiment")
    if st.button("Analyze sentiment"):
        if user_text.strip():
            seq = tokenizer.texts_to_sequences([user_text])
            pad = pad_sequences(seq, maxlen=100)
            pred = model.predict(pad)
            sentiment_idx = np.argmax(pred) #we do this cause of probablities (softmax)
            sentiment = sentiment_labels[sentiment_idx]
            confidence = pred[0][sentiment_idx]
            st.success(f"Predicted Sentiment: {sentiment}: ({confidence:.2%}% confidence) ")



sentiment_model()

