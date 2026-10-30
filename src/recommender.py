import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
Recommendation System Design
As formulated in Section III-D of the paper:
'The recommendation system utilizes Cosine Similarity to calculate the
similarity between customer sentiment-behavioral vectors and products,
generating context-aware recommendations. Cosine similarity measures
the angle between two vectors, ensuring that even subtle preferences
are factored into recommendations.'
"""

class CosineRecommendationEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.product_catalog = df[['product_id', 'product_name', 'category', 'unit_price']].drop_duplicates('product_id').reset_index(drop=True)
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        self._build_product_representation()
        self._build_user_profiles()

    def _build_product_representation(self):
        """Constructs product vectors integrating metadata and aggregated sentiment."""
        prod_meta = []
        for _, row in self.product_catalog.iterrows():
            prod_reviews = self.df[self.df['product_id'] == row['product_id']]['review_title'].fillna('').tolist()
            text_context = " ".join(prod_reviews[:15])
            meta_str = f"{row['product_name']} {row['category']} {text_context}"
            prod_meta.append(meta_str)

        self.tfidf_matrix = self.tfidf.fit_transform(prod_meta)
        # Compute product-product cosine similarity
        self.product_cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _build_user_profiles(self):
        """Builds customer preference vectors based on past purchases and sentiment scores."""
        if 'sentiment_score' in self.df.columns:
            sent_series = self.df['sentiment_score']
        elif 'vader_compound' in self.df.columns:
            sent_series = self.df['vader_compound']
        else:
            sent_series = (self.df['rating'] - 3.0) / 2.0

        self.df['context_weight'] = self.df['rating'] + (sent_series * 0.5)

        self.user_item_matrix = self.df.pivot_table(
            index='user_id',
            columns='product_id',
            values='context_weight',
            aggfunc='mean'
        ).fillna(0)

        if not self.user_item_matrix.empty and self.user_item_matrix.shape[0] > 1:
            # Pairwise customer-customer cosine similarity
            self.user_cosine_sim = cosine_similarity(self.user_item_matrix)
            self.user_sim_df = pd.DataFrame(self.user_cosine_sim, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        else:
            self.user_sim_df = pd.DataFrame()

    def recommend_for_user(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Generates context-aware product recommendations for a customer using Cosine Similarity."""
        if user_id not in self.user_item_matrix.index:
            return self.get_top_trending(top_n)

        user_history = self.df[self.df['user_id'] == user_id]
        purchased_pids = set(user_history['product_id'])

        sim_scores = self.user_sim_df.loc[user_id].drop(user_id)
        top_similar_users = sim_scores.sort_values(ascending=False).head(10)

        cand_scores = {}
        for sim_user, weight in top_similar_users.items():
            if weight <= 0:
                continue
            sim_user_ratings = self.user_item_matrix.loc[sim_user]
            for pid, rating in sim_user_ratings.items():
                if pid not in purchased_pids and rating > 0:
                    cand_scores[pid] = cand_scores.get(pid, 0.0) + (rating * weight)

        if not cand_scores:
            sorted_history = user_history.sort_values('rating', ascending=False)
            fav_item = sorted_history['product_id'].iloc[0] if not sorted_history.empty else self.product_catalog['product_id'].iloc[0]
            exclude = purchased_pids if len(purchased_pids) < len(self.product_catalog) else {fav_item}
            fallback = self.recommend_similar_products(fav_item, top_n=top_n, exclude_pids=exclude)
            return fallback if not fallback.empty else self.get_top_trending(top_n)

        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        res_pids = [p[0] for p in ranked]

        recs = self.product_catalog[self.product_catalog['product_id'].isin(res_pids)].copy()
        recs['recommendation_score'] = recs['product_id'].map(dict(ranked)).round(2)
        return recs.sort_values('recommendation_score', ascending=False)

    def recommend_similar_products(self, product_id: str, top_n: int = 5, exclude_pids: set = None) -> pd.DataFrame:
        """Generates item-to-item recommendations via Cosine Similarity on content vectors."""
        if product_id not in self.product_catalog['product_id'].values:
            return self.get_top_trending(top_n)

        idx = self.product_catalog[self.product_catalog['product_id'] == product_id].index[0]
        sim_vector = self.product_cosine_sim[idx]
        sim_indices = np.argsort(sim_vector)[::-1]

        recs = []
        for i in sim_indices:
            pid = self.product_catalog.iloc[i]['product_id']
            if pid == product_id or (exclude_pids and pid in exclude_pids):
                continue
            recs.append((pid, round(float(sim_vector[i]), 3)))
            if len(recs) >= top_n:
                break

        res_pids = [r[0] for r in recs]
        res_df = self.product_catalog[self.product_catalog['product_id'].isin(res_pids)].copy()
        res_df['similarity_score'] = res_df['product_id'].map(dict(recs))
        return res_df.sort_values('similarity_score', ascending=False)

    def get_top_trending(self, top_n: int = 5) -> pd.DataFrame:
        """Fallback to top-rated products with positive sentiment."""
        sentiment_col = 'sentiment_score' if 'sentiment_score' in self.df.columns else 'vader_compound'
        stats = self.df.groupby('product_id').agg(
            avg_rating=('rating', 'mean'),
            review_count=('rating', 'count'),
            avg_sentiment=(sentiment_col, 'mean')
        ).reset_index()

        stats['popularity_score'] = (stats['avg_rating'] * 0.6) + (stats['avg_sentiment'] * 2.0)
        top_stats = stats.sort_values('popularity_score', ascending=False).head(top_n)
        return pd.merge(top_stats, self.product_catalog, on='product_id')[
            ['product_id', 'product_name', 'category', 'unit_price', 'avg_rating', 'review_count']
        ]

# Alias for backward compatibility with app.py imports
HybridRecommender = CosineRecommendationEngine
