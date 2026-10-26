import re
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

try:
    sia = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    """Preprocess text: lowercase, remove URLs, special characters, and whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text: str, rating: float = None) -> dict:
    """Computes VADER & TextBlob sentiment scores and labels."""
    cleaned = clean_text(text)
    if not cleaned:
        if rating is not None:
            if rating >= 4:
                return {"vader_compound": 0.5, "sentiment": "Positive", "polarity": 0.5, "subjectivity": 0.5, "vader_pos": 0.5, "vader_neu": 0.5, "vader_neg": 0.0}
            elif rating <= 2:
                return {"vader_compound": -0.5, "sentiment": "Negative", "polarity": -0.5, "subjectivity": 0.5, "vader_pos": 0.0, "vader_neu": 0.5, "vader_neg": 0.5}
            else:
                return {"vader_compound": 0.0, "sentiment": "Neutral", "polarity": 0.0, "subjectivity": 0.5, "vader_pos": 0.1, "vader_neu": 0.8, "vader_neg": 0.1}
        return {"vader_compound": 0.0, "sentiment": "Neutral", "polarity": 0.0, "subjectivity": 0.0, "vader_pos": 0.0, "vader_neu": 1.0, "vader_neg": 0.0}

    vader_scores = sia.polarity_scores(cleaned)
    compound = vader_scores['compound']
    blob = TextBlob(cleaned)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        if rating and rating >= 4:
            sentiment = "Positive"
        elif rating and rating <= 2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    return {
        "vader_pos": round(vader_scores.get('pos', 0.0), 3),
        "vader_neu": round(vader_scores.get('neu', 0.0), 3),
        "vader_neg": round(vader_scores.get('neg', 0.0), 3),
        "vader_compound": round(compound, 4),
        "polarity": round(tb_polarity, 4),
        "subjectivity": round(tb_subjectivity, 4),
        "sentiment": sentiment
    }

def process_sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches reviews dataframe with sentiment scores."""
    df_copy = df.copy()
    results = []
    for _, row in df_copy.iterrows():
        title = str(row.get('review_title', '')) if pd.notnull(row.get('review_title')) else ''
        text = str(row.get('review_text', '')) if pd.notnull(row.get('review_text')) else ''
        combined = f"{title} {text}".strip()
        rating = row.get('rating', None)
        scores = analyze_sentiment(combined, rating)
        results.append(scores)
    
    scores_df = pd.DataFrame(results)
    for col in scores_df.columns:
        df_copy[col] = scores_df[col].values
    return df_copy
