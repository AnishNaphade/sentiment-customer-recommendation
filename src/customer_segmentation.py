import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import joblib
import os

"""
Customer Segmentation Module
As formulated in Section III-C of the paper:
'We use two primary machine learning techniques for segmentation:
K-means clustering and K-nearest neighbours (KNN).
The K-means algorithm groups customers based on the similarity of their
behaviours and sentiments, selecting optimal initial centroids (via Elbow method).
KNN is used to classify new incoming customers into appropriate segments.'
"""

class CustomerSegmenter:
    def __init__(self, n_clusters: int = 3):
        # As determined by Elbow Method in the research paper (Figure 2: 3 clusters: 0, 1, 2)
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.cluster_labels_map = {
            0: "High-Engagement / Positive Sentiment Buyers",
            1: "Moderate / Neutral Transactional Shoppers",
            2: "Critical / At-Risk Reviewers"
        }

    def build_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts transactional and sentiment-derived behavioral vectors per customer:
        - transaction_volume (Invoice frequency)
        - monetary_value (Total spend)
        - sentiment_profile (Continuous Stanford CoreNLP sentiment score)
        - average_rating (Mean user feedback)
        """
        df_clean = df.copy()

        sent_col = 'sentiment_score' if 'sentiment_score' in df_clean.columns else ('vader_compound' if 'vader_compound' in df_clean.columns else 'rating')
        agg_dict = {
            'reviewer_name': ('reviewer_name', 'first'),
            'country': ('country', 'first'),
            'transaction_volume': ('review_id', 'count'),
            'monetary_value': ('total_spend', 'sum'),
            'average_rating': ('rating', 'mean')
        }
        if sent_col == 'rating':
            agg_dict['sentiment_score'] = ('rating', lambda r: (r.mean() - 3.0) / 2.0)
        else:
            agg_dict['sentiment_score'] = (sent_col, 'mean')

        user_group = df_clean.groupby('user_id').agg(**agg_dict).reset_index()

        user_group['monetary_value'] = user_group['monetary_value'].round(2)
        user_group['average_rating'] = user_group['average_rating'].round(2)
        user_group['sentiment_score'] = user_group['sentiment_score'].round(3)

        # Retain backward-compatible fields for UI
        user_group['recency'] = 15
        user_group['frequency'] = user_group['transaction_volume']
        user_group['monetary'] = user_group['monetary_value']
        user_group['avg_rating'] = user_group['average_rating']
        user_group['avg_sentiment'] = user_group['sentiment_score']

        return user_group

    def fit_predict(self, features_df: pd.DataFrame):
        """
        Stage 1: K-Means clustering to discover primary personas.
        Stage 2: Fit KNN classifier on cluster assignments to enable dynamic adaptation for new users.
        """
        feature_cols = ['transaction_volume', 'monetary_value', 'average_rating', 'sentiment_score']
        X = features_df[feature_cols].copy()

        # Scale feature space
        scaled_X = self.scaler.fit_transform(X)
        
        # Fit K-Means
        clusters = self.kmeans.fit_predict(scaled_X)
        sil_score = silhouette_score(scaled_X, clusters) if len(set(clusters)) > 1 else 0.0

        # Fit KNN on clusters as specified in Section III-C
        self.knn.fit(scaled_X, clusters)

        features_df['cluster'] = clusters
        features_df['segment_name'] = features_df['cluster'].map(self.cluster_labels_map)

        return features_df, sil_score

    def classify_new_customer(self, new_customer_features: dict) -> int:
        """Uses KNN to assign a new customer to the closest established behavioral segment."""
        raw = np.array([[
            new_customer_features.get('transaction_volume', 1),
            new_customer_features.get('monetary_value', 50.0),
            new_customer_features.get('average_rating', 3.0),
            new_customer_features.get('sentiment_score', 0.0)
        ]])
        scaled = self.scaler.transform(raw)
        return int(self.knn.predict(scaled)[0])

    def save_model(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.kmeans, os.path.join(model_dir, "kmeans_model.joblib"))
        joblib.dump(self.knn, os.path.join(model_dir, "knn_model.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        joblib.dump(self.cluster_labels_map, os.path.join(model_dir, "labels_map.joblib"))
