import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
import pickle
import seaborn as sns
import requests



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
