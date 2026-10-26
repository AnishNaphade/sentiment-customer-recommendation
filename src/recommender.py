import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.product_catalog = df[['product_id', 'product_name', 'category', 'unit_price']].drop_duplicates('product_id').reset_index(drop=True)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self._build_content_matrix()
        self._build_collaborative_matrix()

    def _build_content_matrix(self):
        # Metadata description for products
        prod_meta = []
        for _, row in self.product_catalog.iterrows():
            # Gather associated review text to enrich content metadata
            prod_reviews = self.df[self.df['product_id'] == row['product_id']]['review_title'].fillna('').tolist()
            review_snippets = " ".join(prod_reviews[:15])
            meta_str = f"{row['product_name']} {row['category']} {review_snippets}"
            prod_meta.append(meta_str)

        self.tfidf_matrix = self.tfidf.fit_transform(prod_meta)
        self.content_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _build_collaborative_matrix(self):
        # User-Item interaction matrix (Rating weighted by Sentiment)
        self.interaction_df = self.df.copy()
        if 'vader_compound' in self.interaction_df.columns:
            # Weight rating by sentiment (+0.5 for positive, -0.5 for negative)
            self.interaction_df['adjusted_rating'] = self.interaction_df['rating'] + (self.interaction_df['vader_compound'] * 0.5)
        else:
            self.interaction_df['adjusted_rating'] = self.interaction_df['rating']

        self.user_item_matrix = self.interaction_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='adjusted_rating',
            aggfunc='mean'
        ).fillna(0)

        if not self.user_item_matrix.empty and self.user_item_matrix.shape[0] > 1:
            self.user_sim = cosine_similarity(self.user_item_matrix)
            self.user_sim_df = pd.DataFrame(self.user_sim, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        else:
            self.user_sim_df = pd.DataFrame()

    def recommend_for_user(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Collaborative & Sentiment-adjusted recommendation for a given customer."""
        if user_id not in self.user_item_matrix.index:
            # Fallback to top-rated / highest-sentiment products
            return self.get_top_trending(top_n)

        # Products user has already bought/reviewed
        user_history = self.interaction_df[self.interaction_df['user_id'] == user_id]
        purchased_pids = set(user_history['product_id'])

        # Find most similar users
        sim_scores = self.user_sim_df.loc[user_id].drop(user_id)
        top_similar_users = sim_scores.sort_values(ascending=False).head(10)

        # Candidate item scores
        cand_scores = {}
        for sim_user, weight in top_similar_users.items():
            if weight <= 0:
                continue
            sim_user_ratings = self.user_item_matrix.loc[sim_user]
            for pid, rating in sim_user_ratings.items():
                if pid not in purchased_pids and rating > 0:
                    cand_scores[pid] = cand_scores.get(pid, 0.0) + (rating * weight)

        if not cand_scores:
            # Content fallback based on user's highest rated item
            sorted_history = user_history.sort_values('rating', ascending=False)
            fav_item = sorted_history['product_id'].iloc[0] if not sorted_history.empty else self.product_catalog['product_id'].iloc[0]
            # If the user has purchased all/most products, do not exclude all items; exclude at most the anchor item
            exclude = purchased_pids if len(purchased_pids) < len(self.product_catalog) else {fav_item}
            fallback_recs = self.recommend_similar_products(fav_item, top_n=top_n, exclude_pids=exclude)
            if not fallback_recs.empty:
                return fallback_recs
            return self.get_top_trending(top_n)

        # Rank items
        ranked_pids = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        res_pids = [p[0] for p in ranked_pids]

        recs = self.product_catalog[self.product_catalog['product_id'].isin(res_pids)].copy()
        recs['recommendation_score'] = recs['product_id'].map(dict(ranked_pids)).round(2)
        recs = recs.sort_values('recommendation_score', ascending=False)
        if recs.empty:
            return self.get_top_trending(top_n)
        return recs

    def recommend_similar_products(self, product_id: str, top_n: int = 5, exclude_pids: set = None) -> pd.DataFrame:
        """Content-based recommendation based on product metadata & textual reviews."""
        if product_id not in self.product_catalog['product_id'].values:
            return self.get_top_trending(top_n)

        idx = self.product_catalog[self.product_catalog['product_id'] == product_id].index[0]
        sim_vector = self.content_sim[idx]

        sim_indices = np.argsort(sim_vector)[::-1]
        recs = []
        for i in sim_indices:
            pid = self.product_catalog.iloc[i]['product_id']
            if pid == product_id or (exclude_pids and pid in exclude_pids):
                continue
            recs.append((pid, round(float(sim_vector[i]), 3)))
            if len(recs) >= top_n:
                break

        # If exclusion was too strict and left no items, relax exclusion to just product_id itself
        if not recs:
            for i in sim_indices:
                pid = self.product_catalog.iloc[i]['product_id']
                if pid == product_id:
                    continue
                recs.append((pid, round(float(sim_vector[i]), 3)))
                if len(recs) >= top_n:
                    break

        if not recs:
            return self.get_top_trending(top_n)

        res_pids = [r[0] for r in recs]
        sim_scores_map = dict(recs)
        result_df = self.product_catalog[self.product_catalog['product_id'].isin(res_pids)].copy()
        result_df['similarity_score'] = result_df['product_id'].map(sim_scores_map)
        return result_df.sort_values('similarity_score', ascending=False)

    def get_top_trending(self, top_n: int = 5) -> pd.DataFrame:
        """Returns top products rated with positive sentiment."""
        stats = self.df.groupby('product_id').agg(
            avg_rating=('rating', 'mean'),
            review_count=('rating', 'count'),
            avg_compound=('vader_compound', 'mean') if 'vader_compound' in self.df.columns else ('rating', 'mean')
        ).reset_index()

        stats['popularity_score'] = (stats['avg_rating'] * 0.6) + (stats['avg_compound'] * 2.0) + (np.log1p(stats['review_count']) * 0.5)
        top_stats = stats.sort_values('popularity_score', ascending=False).head(top_n)
        
        merged = pd.merge(top_stats, self.product_catalog, on='product_id')
        return merged[['product_id', 'product_name', 'category', 'unit_price', 'avg_rating', 'review_count']]
