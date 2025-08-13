import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import requests
import ollama

class SentimentAnalyzer:
    def __init__(self) -> None:
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.best_model = None

    def load_sample_data(self):
        """ Load sample data for training """
        sample_data = {
            'text': [
                'I love this product, it works perfectly!',
                'This is terrible, waste of money',
                'Average product, nothing special',
                'Excellent quality, highly recommend',
                'Poor customer service, very disappointed',
                'Good value for money',
                'Not what I expected, could be better',
                'Amazing experience, will buy again',
                'Decent product, meets expectations',
                'Horrible quality, avoid this'
            ],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                         'positive', 'negative', 'positive', 'neutral', 'negative']
        }
        return pd.DataFrame(sample_data)
    
    def train_models(self, df):
        """ Train multiple models on the dataset """
        X = df['text']
        y = df['sentiment']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', random_state=42)
        }

        best_score = 0
        best_model_name = None

        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', model)
            ])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Predict on the evaluation set
            y_pred = pipeline.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            print(f"{name} Accuracy: {score:.4f}")

            self.models[name] = pipeline

            if score > best_score:
                best_score = score
                best_model_name = name
            
        self.best_model = self.models[best_model_name]
        print(f"Best model: {best_model_name} with accuracy {best_score:.4f}")

        return self.best_model

    def predict_sentimental_local(self, text):
        """ Predict sentiment using the best local model """
        if not self.best_model:
            raise ValueError("No model trained yet. Please train the model first.")
        
        prediction = self.best_model.predict([text])[0]
        probablities = self.best_model.predict_proba([text])[0]
        
        return {
            'sentiment': prediction,
            'confidence': max(probablities),
            'probabilities': dict(zip(self.best_model.classes_, probablities))
        }
    
    def predict_sentimental_ollama(self, text, model_name='llama3.2:1b'):
        """ Predict sentiment using Ollama API """
        if not self.best_model:
            raise ValueError("No model trained yet. Please train the model first.")

        try:
            prompt = f"""
                Analyze the sentiment of this text and respond with only one word: positive, negative, or neutral.
                Text: {text}
                sentiment:
                """
            response = ollama.generate(model=model_name, prompt=prompt)

            # Ensure the sentiment is one of the expected values
            sentiment = response['text'].strip().lower()
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'  # Default to neutral if unexpected response

            return {
                'sentiment': sentiment,
                'model': model_name,
                'method': 'ollama'
            }
        
        except Exception as e:
            print(f"Error during Ollama prediction: {e}")
            return {
                'sentiment': 'neutral',
                'model': model_name,
                'method': 'ollama',
                'error': str(e)
            }
        def save_model(self, filename='sentiment_model.pkl'):
            """ Save the best model to a file """
            if not self.best_model:
                raise ValueError("No model trained yet. Please train the model first.")
            
            joblib.dump(self.best_model, filename)
            print(f"Model saved to {filename}")

            
    def load_model(self, filename='sentiment_model.pkl'):
        """ Load a model from a file """
        try:
            self.best_model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"Model file {filename} not found.")
            self.best_model = None
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.best_model = None
            return None
        return self.best_model
    

