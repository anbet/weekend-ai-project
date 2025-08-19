import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ...existing code...

from src.data_preprocessing import TextPreprocessor
from src.sentiment_model import SentimentAnalyzer

# Initialisation of the preprocessor and sentiment analyzer
preprocessor = TextPreprocessor()
sentiment_analyzer = SentimentAnalyzer()

# Load and train sample data
df = sentiment_analyzer.load_sample_data()
model = sentiment_analyzer.train_models(df)

# Test prediction with sample data
sample_texts = [
    "I love sunny days and beautiful weather!",
    "I hate getting stuck in traffic jams.",
    "The book is on the table."
]

# Print sentiment analysis results
for text in sample_texts:
    local_result = sentiment_analyzer.predict_sentimental_local(text)
    ollama_result = sentiment_analyzer.predict_sentimental_ollama(text)
    print(f"Text: {text}")
    print(f"Local Model Prediction: {local_result}")
    print(f"Ollama Model Prediction: {ollama_result}")
    print("-" * 50)

