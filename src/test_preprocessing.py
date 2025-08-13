import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import TextPreprocessor

# src/test_data_preprocessing.py


@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_get_sentiment_textblob_positive(preprocessor):
    text = "I love sunny days and beautiful weather!"
    result = preprocessor.get_sentiment_textblob(text)
    assert 'polarity' in result
    assert 'subjectivity' in result
    assert 'sentiment' in result
    assert result['sentiment'] == 'Positive'

def test_get_sentiment_textblob_negative(preprocessor):
    text = "I hate getting stuck in traffic jams."
    result = preprocessor.get_sentiment_textblob(text)
    assert result['sentiment'] == 'Negative'

def test_get_sentiment_textblob_neutral(preprocessor):
    text = "The book is on the table."
    result = preprocessor.get_sentiment_textblob(text)
    assert result['sentiment'] == 'Neutral'