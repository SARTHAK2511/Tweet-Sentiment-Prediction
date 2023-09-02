
import streamlit as st
import joblib
from nltk.tokenize import TweetTokenizer
import re

# Load your trained model
logprior, loglikelihood = joblib.load(r'C:\Users\HP\Downloads\sentiment_model.joblib')

# Function to preprocess and classify sentiment
def classify_sentiment(tweet, logprior, loglikelihood):
    def lookup(freqs, word, label):
        if (word, label) in freqs:
            return freqs[(word, label)]
        else:
            return 0

    def naive_bayes_predict(tweet, logprior, loglikelihood):
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        p = 0
        p += logprior
        for word in tweet_tokens:
            if word in loglikelihood:
                p += loglikelihood[word]
        return p

    sentiment_score = naive_bayes_predict(tweet, logprior, loglikelihood)
    return "Positive" if sentiment_score > 0 else "Negative"

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a tweet to analyze its sentiment:")

# User input
tweet_input = st.text_input("Enter a tweet:")

if st.button("Analyze"):
    if tweet_input:
        # Preprocess the tweet
        tweet_input = re.sub(r'\$\w*', '', tweet_input)
        tweet_input = re.sub(r'^RT[\s]+', '', tweet_input)
        tweet_input = re.sub(r'https?://[^\s\n\r]+', '', tweet_input)
        tweet_input = re.sub(r'#', '', tweet_input)

        # Classify sentiment
        sentiment = classify_sentiment(tweet_input, logprior, loglikelihood)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a tweet for analysis.")


st.markdown("#### Connect with me:")
st.markdown("[GitHub](https://github.com/SARTHAK2511/)")
st.markdown("[LinkedIn](https://www.linkedin.com/in/sarthak-bhatore-004aaa1ba/)")

# Add your name
st.markdown("#### Created by: Sarthak Bhatore")
