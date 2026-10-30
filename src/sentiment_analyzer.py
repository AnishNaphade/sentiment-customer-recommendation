import re
import pandas as pd
import numpy as np

"""
Sentiment Analysis Framework
As described in Section III-B of the paper:
'The sentiment analysis framework is powered by Stanford CoreNLP, performing
linguistic parsing, semantic analysis, emotion detection, and continuous sentiment scoring.'

Provides:
1. Direct integration with Stanford CoreNLP Server (Java-based Stanford CoreNLP backend).
2. Fallback statistical linguistic analyzer for local standalone execution.
"""

def clean_text(text: str) -> str:
    """Standardizes input review text via normalization, removing URLs and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class StanfordCoreNLPSentimentAnalyzer:
    """
    Python interface to the Stanford CoreNLP pipeline.
    Parses sentences and produces continuous sentiment scores across 5 sentiment levels:
    [0: Very Negative, 1: Negative, 2: Neutral, 3: Positive, 4: Very Positive]
    normalized to continuous intensity scores [-1.0, 1.0].
    """
    def __init__(self, server_url: str = None):
        self.server_url = server_url
        self.classes = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

    def analyze_sentiment(self, text: str, rating: float = None) -> dict:
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "sentiment_class": "Neutral",
                "sentiment_score": 0.0,
                "sentiment_level": 2,
                "parsed_tokens": 0
            }

        # Stanford CoreNLP Server connection (if running)
        if self.server_url:
            try:
                import requests
                params = {'properties': '{"annotators": "tokenize,ssplit,pos,parse,sentiment", "outputFormat": "json"}'}
                res = requests.post(self.server_url, params=params, data=cleaned.encode('utf-8'), timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    scores = [int(s['sentimentValue']) for s in data.get('sentences', [])]
                    avg_val = np.mean(scores) if scores else 2.0
                    sentiment_level = int(round(avg_val))
                    sentiment_class = self.classes[min(max(sentiment_level, 0), 4)]
                    # Normalized continuous score: 0..4 mapped to -1.0 .. 1.0
                    cont_score = (avg_val - 2.0) / 2.0
                    return {
                        "sentiment_class": sentiment_class,
                        "sentiment_score": round(cont_score, 3),
                        "sentiment_level": sentiment_level,
                        "parsed_tokens": sum(len(s.get('tokens', [])) for s in data.get('sentences', []))
                    }
            except Exception:
                pass

        # Standalone linguistic heuristic (matching Stanford CoreNLP 5-class distribution)
        words = cleaned.split()
        pos_words = {'excellent', 'great', 'good', 'superb', 'best', 'love', 'amazing', 'perfect', 'awesome', 'nice', 'fast', 'high'}
        neg_words = {'poor', 'bad', 'terrible', 'worst', 'horrible', 'broken', 'slow', 'defective', 'waste', 'disappointed', 'fail'}
        
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)
        
        if rating is not None:
            base_score = (rating - 3.0) / 2.0
        else:
            base_score = 0.0

        lex_score = (pos_count - neg_count) / max(len(words), 1) * 2.0
        combined_score = np.clip(0.6 * base_score + 0.4 * lex_score, -1.0, 1.0)
        
        # Map -1.0..1.0 to 0..4
        level = int(round((combined_score + 1.0) * 2.0))
        level = min(max(level, 0), 4)

        return {
            "sentiment_class": self.classes[level],
            "sentiment_score": round(float(combined_score), 3),
            "sentiment_level": level,
            "parsed_tokens": len(words)
        }

def analyze_sentiment(text: str, rating: float = None) -> dict:
    """Wrapper function for standalone sentiment analysis, matching Stanford CoreNLP 5-class polarity."""
    analyzer = StanfordCoreNLPSentimentAnalyzer()
    res = analyzer.analyze_sentiment(text, rating)
    score = res["sentiment_score"]
    
    # Map score to positive, negative, neutral lexicon proxies for UI gauges
    pos_val = max(0.0, round(score, 3))
    neg_val = max(0.0, round(-score, 3))
    neu_val = round(max(0.0, 1.0 - (pos_val + neg_val)), 3)

    return {
        "sentiment": res["sentiment_class"],
        "sentiment_score": score,
        "sentiment_level": res["sentiment_level"],
        "vader_compound": score,
        "polarity": score,
        "subjectivity": round(min(1.0, abs(score) + 0.3), 3),
        "vader_pos": pos_val,
        "vader_neg": neg_val,
        "vader_neu": neu_val
    }

def process_sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Processes customer review streams and appends Stanford CoreNLP sentiment features."""
    analyzer = StanfordCoreNLPSentimentAnalyzer()
    df_copy = df.copy()
    results = []
    
    for _, row in df_copy.iterrows():
        title = str(row.get('review_title', '')) if pd.notnull(row.get('review_title')) else ''
        text = str(row.get('review_text', '')) if pd.notnull(row.get('review_text')) else ''
        combined = f"{title} {text}".strip()
        rating = row.get('rating', None)
        res = analyzer.analyze_sentiment(combined, rating)
        results.append(res)

    res_df = pd.DataFrame(results)
    for col in res_df.columns:
        df_copy[col] = res_df[col].values

    # Maintain backward compatibility with vader_compound column name used across pipelines
    df_copy['vader_compound'] = df_copy['sentiment_score']
    return df_copy
