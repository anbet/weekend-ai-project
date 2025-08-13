import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str):
        """ Clean and pre-process text """
        # Convert text to lowercase
        text = text.lower()

        # Remove urls, mentions and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters and digits
        text = re.sub(r'^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_sentiment_textblob(self, text):
        """ Get sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity # type: ignore
        subjectivity = blob.sentiment.subjectivity # type: ignore
        sentiment = {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': 'Positive' if polarity > 0.1
                        else 'Negative' if polarity < -0.1
                        else 'Neutral'
        }
        return sentiment
