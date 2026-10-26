import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

class CustomerSegmenter:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels_map = {}

    def build_rfm_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates:
        - Recency: Days since most recent review/transaction
        - Frequency: Total transactions / reviews
        - Monetary: Total monetary spend
        - Avg_Rating: Mean star rating given
        - Avg_Sentiment: Mean VADER compound score
        - Helpful_Votes: Total helpful feedback received
        """
        df_clean = df.copy()
        df_clean['review_date'] = pd.to_datetime(df_clean['review_date'])
        reference_date = df_clean['review_date'].max() + pd.Timedelta(days=1)

        user_group = df_clean.groupby('user_id').agg(
            reviewer_name=('reviewer_name', 'first'),
            country=('country', 'first'),
            recency=('review_date', lambda dates: (reference_date - dates.max()).days),
            frequency=('review_id', 'count'),
            monetary=('total_spend', 'sum'),
            avg_rating=('rating', 'mean'),
            avg_sentiment=('vader_compound', 'mean') if 'vader_compound' in df_clean.columns else ('rating', lambda r: (r.mean() - 3) / 2),
            total_helpful=('helpful_votes', 'sum') if 'helpful_votes' in df_clean.columns else ('rating', 'count')
        ).reset_index()

        user_group['monetary'] = user_group['monetary'].round(2)
        user_group['avg_rating'] = user_group['avg_rating'].round(2)
        user_group['avg_sentiment'] = user_group['avg_sentiment'].round(3)

        return user_group

    def fit_predict(self, features_df: pd.DataFrame):
        feature_cols = ['recency', 'frequency', 'monetary', 'avg_rating', 'avg_sentiment']
        X = features_df[feature_cols].copy()

        # Handle log transform for skewed metrics (Frequency & Monetary)
        X['frequency_log'] = np.log1p(X['frequency'])
        X['monetary_log'] = np.log1p(X['monetary'])
        fit_cols = ['recency', 'frequency_log', 'monetary_log', 'avg_rating', 'avg_sentiment']

        scaled_X = self.scaler.fit_transform(X[fit_cols])
        clusters = self.model.fit_predict(scaled_X)
        
        sil_score = silhouette_score(scaled_X, clusters) if len(set(clusters)) > 1 else 0.0

        features_df['cluster'] = clusters
        
        # Name the clusters based on profile averages
        cluster_summary = features_df.groupby('cluster').agg({
            'monetary': 'mean',
            'frequency': 'mean',
            'recency': 'mean',
            'avg_sentiment': 'mean'
        })

        # Heuristic mapping for intuitive business persona tags
        for c in range(self.n_clusters):
            row = cluster_summary.loc[c]
            if row['monetary'] >= cluster_summary['monetary'].median() and row['avg_sentiment'] >= 0.2:
                self.cluster_labels_map[c] = "Champions & Brand Advocates"
            elif row['monetary'] >= cluster_summary['monetary'].median() and row['avg_sentiment'] < 0.2:
                self.cluster_labels_map[c] = "High-Value At-Risk Buyers"
            elif row['recency'] > cluster_summary['recency'].median():
                self.cluster_labels_map[c] = "Hibernating / Dormant Shoppers"
            else:
                self.cluster_labels_map[c] = "Occasional / Growth Prospects"

        features_df['segment_name'] = features_df['cluster'].map(self.cluster_labels_map)
        return features_df, sil_score

    def save_model(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "kmeans_model.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        joblib.dump(self.cluster_labels_map, os.path.join(model_dir, "labels_map.joblib"))
