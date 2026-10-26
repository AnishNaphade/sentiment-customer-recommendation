import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Ensure src modules are resolvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sentiment_analyzer import analyze_sentiment, process_sentiment_dataframe
from src.customer_segmentation import CustomerSegmenter
from src.recommender import HybridRecommender

# Page Configuration
st.set_page_config(
    page_title="AI Consumer Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Sleek Dark Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.3rem;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .main-header p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .badge-positive {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-neutral {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Data Loader with Caching
@st.cache_data
def load_and_prepare_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_path = os.path.join(base_dir, "data", "amazon_reviews.csv")
    scored_path = os.path.join(base_dir, "data", "reviews_with_sentiment.csv")
    segmented_path = os.path.join(base_dir, "data", "customer_segments.csv")

    if os.path.exists(scored_path) and os.path.exists(segmented_path):
        df = pd.read_csv(scored_path)
        seg_df = pd.read_csv(segmented_path)
    else:
        df = pd.read_csv(raw_path)
        df = process_sentiment_dataframe(df)
        df.to_csv(scored_path, index=False)

        segmenter = CustomerSegmenter(n_clusters=4)
        rfm_df = segmenter.build_rfm_sentiment_features(df)
        seg_df, _ = segmenter.fit_predict(rfm_df)
        seg_df.to_csv(segmented_path, index=False)

    return df, seg_df

@st.cache_resource
def get_recommender(df):
    return HybridRecommender(df)

df_reviews, df_segments = load_and_prepare_data()
recommender = get_recommender(df_reviews)

# Sidebar Navigation & Filters
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4a/Amazon_icon.svg", width=50)
    st.title("Navigation")
    app_mode = st.radio(
        "Modules",
        [
            "Executive Dashboard",
            "Sentiment Deep-Dive",
            "Live Text Analyzer",
            "Customer Segmentation",
            "Recommendation Engine",
            "Raw Data Explorer"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Global Filters")
    selected_country = st.selectbox("Market Region", ["All Countries"] + sorted(list(df_reviews['country'].unique())))
    selected_cat = st.selectbox("Category", ["All Categories"] + sorted(list(df_reviews['category'].unique())))

# Filter data according to selection
filtered_df = df_reviews.copy()
if selected_country != "All Countries":
    filtered_df = filtered_df[filtered_df['country'] == selected_country]
if selected_cat != "All Categories":
    filtered_df = filtered_df[filtered_df['category'] == selected_cat]

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>E-Commerce Intelligence Platform</h1>
    <p>Unified NLP Sentiment Analysis, Dynamic RFM+Sentiment Customer Segmentation & Hybrid Recommendation Engine</p>
</div>
""", unsafe_allow_html=True)

# ----------------- MODULE 1: EXECUTIVE DASHBOARD -----------------
if app_mode == "Executive Dashboard":
    st.subheader("Key Performance Indicators")
    
    total_reviews = len(filtered_df)
    unique_customers = filtered_df['user_id'].nunique()
    total_gmv = filtered_df['total_spend'].sum()
    avg_rating = filtered_df['rating'].mean()
    pos_sentiment_pct = (filtered_df['sentiment'] == 'Positive').mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Reviews</div>
            <div class="metric-val">{total_reviews:,}</div>
            <div class="metric-sub" style="color:#10b981;">Across 20 Products</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Shoppers</div>
            <div class="metric-val">{unique_customers:,}</div>
            <div class="metric-sub" style="color:#60a5fa;">8 Global Regions</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Catalog GMV</div>
            <div class="metric-val">${total_gmv:,.0f}</div>
            <div class="metric-sub" style="color:#a78bfa;">Avg ${total_gmv/max(1,total_reviews):.1f}/order</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Star Rating</div>
            <div class="metric-val">{avg_rating:.2f} ⭐</div>
            <div class="metric-sub" style="color:#f59e0b;">Scale 1.0 - 5.0</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Positive Sent.</div>
            <div class="metric-val">{pos_sentiment_pct:.1f}%</div>
            <div class="metric-sub" style="color:#34d399;">VADER & Polarity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Analytics Row
    row1_c1, row1_c2 = st.columns([1.2, 1])
    
    with row1_c1:
        st.markdown("##### Monthly Rating & Sentiment Trajectory")
        monthly = filtered_df.copy()
        monthly['month_year'] = pd.to_datetime(monthly['review_date']).dt.to_period('M').astype(str)
        monthly_stats = monthly.groupby('month_year').agg(
            avg_rating=('rating', 'mean'),
            avg_compound=('vader_compound', 'mean'),
            volume=('review_id', 'count')
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_stats['month_year'], y=monthly_stats['avg_rating'],
            mode='lines+markers', name='Avg Rating (1-5)',
            line=dict(color='#38bdf8', width=3),
            marker=dict(size=6)
        ))
        fig_trend.add_trace(go.Bar(
            x=monthly_stats['month_year'], y=monthly_stats['avg_compound'],
            name='Sentiment Compound (-1 to +1)',
            marker_color='#818cf8', opacity=0.4,
            yaxis='y2'
        ))
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Star Rating"),
            yaxis2=dict(title="Compound Sentiment", overlaying='y', side='right', range=[-1, 1])
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with row1_c2:
        st.markdown("##### Sentiment Share by Category")
        cat_sentiment = filtered_df.groupby(['category', 'sentiment']).size().reset_index(name='count')
        fig_cat = px.bar(
            cat_sentiment, x='count', y='category', color='sentiment',
            orientation='h',
            color_discrete_map={'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'},
            template="plotly_dark"
        )
        fig_cat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Global breakdown
    st.markdown("##### Regional Performance & Sentiment Health")
    country_summary = filtered_df.groupby('country').agg(
        total_orders=('review_id', 'count'),
        total_sales=('total_spend', 'sum'),
        avg_rating=('rating', 'mean'),
        positive_rate=('sentiment', lambda s: round((s == 'Positive').mean() * 100, 1))
    ).reset_index().sort_values('total_sales', ascending=False)
    
    st.dataframe(
        country_summary.style.format({
            "total_sales": "${:,.2f}",
            "avg_rating": "{:.2f} ⭐",
            "positive_rate": "{:.1f}%"
        }),
        use_container_width=True
    )

# ----------------- MODULE 2: SENTIMENT DEEP-DIVE -----------------
elif app_mode == "Sentiment Deep-Dive":
    st.subheader("NLP Sentiment Analysis & Lexicon Explorer")
    
    s_col1, s_col2 = st.columns([1, 1])
    
    with s_col1:
        st.markdown("##### Sentiment Distribution Breakdown")
        sent_counts = filtered_df['sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Reviews']
        fig_pie = px.pie(
            sent_counts, values='Reviews', names='Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'},
            hole=0.45,
            template="plotly_dark"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with s_col2:
        st.markdown("##### Polarity vs. Subjectivity Scatter")
        sample_reviews = filtered_df.sample(min(400, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_reviews, x='polarity', y='subjectivity', color='sentiment',
            size='rating', hover_data=['product_name', 'rating'],
            color_discrete_map={'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'},
            template="plotly_dark"
        )
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Word Frequency / Themes
    st.markdown("##### Product-Level Sentiment Matrix")
    prod_sent = filtered_df.groupby('product_name').agg(
        avg_rating=('rating', 'mean'),
        avg_compound=('vader_compound', 'mean'),
        reviews=('review_id', 'count'),
        positive_pct=('sentiment', lambda s: (s == 'Positive').mean() * 100),
        negative_pct=('sentiment', lambda s: (s == 'Negative').mean() * 100)
    ).reset_index().sort_values('avg_compound', ascending=False)

    st.dataframe(
        prod_sent.style.format({
            "avg_rating": "{:.2f} ⭐",
            "avg_compound": "{:.3f}",
            "positive_pct": "{:.1f}%",
            "negative_pct": "{:.1f}%"
        }),
        use_container_width=True
    )

# ----------------- MODULE 3: LIVE TEXT ANALYZER -----------------
elif app_mode == "Live Text Analyzer":
    st.subheader("Real-Time Review Sentiment Diagnostic")
    st.write("Type or paste any consumer feedback or product review below to run through the VADER & TextBlob NLP pipeline in real-time.")

    col_in, col_out = st.columns([1.2, 1])

    with col_in:
        default_sample = "I absolutely love this device! Battery backup lasts for days and setup took only 2 minutes. Outstanding quality."
        user_input_text = st.text_area("Review Content", value=default_sample, height=140)
        optional_rating = st.slider("Customer Rating (Optional)", min_value=1, max_value=5, value=5)
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_btn or user_input_text:
        res = analyze_sentiment(user_input_text, optional_rating)
        
        with col_out:
            st.markdown("##### Inference Results")
            sent = res['sentiment']
            badge_class = "badge-positive" if sent == "Positive" else ("badge-negative" if sent == "Negative" else "badge-neutral")
            
            st.markdown(f"### Classification: <span class='{badge_class}'>{sent.upper()}</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("VADER Compound", f"{res['vader_compound']}", help="-1.0 to +1.0 scale")
                st.metric("TextBlob Polarity", f"{res['polarity']}", help="-1.0 to +1.0 scale")
            with m2:
                st.metric("Subjectivity", f"{res['subjectivity']}", help="0.0 (Objective) to 1.0 (Opinionated)")
                st.metric("Pos / Neg Lexicon", f"{res['vader_pos']} / {res['vader_neg']}")

            # Gauge Chart for Sentiment Intensity
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['vader_compound'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentiment Index", 'font': {'size': 16, 'color': '#cbd5e1'}},
                gauge={
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#38bdf8"},
                    'steps': [
                        {'range': [-1, -0.05], 'color': "rgba(239, 68, 68, 0.3)"},
                        {'range': [-0.05, 0.05], 'color': "rgba(234, 179, 8, 0.3)"},
                        {'range': [0.05, 1], 'color': "rgba(16, 185, 129, 0.3)"}
                    ]
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

# ----------------- MODULE 4: CUSTOMER SEGMENTATION -----------------
elif app_mode == "Customer Segmentation":
    st.subheader("Machine Learning Customer Segmentation (K-Means & RFM-S)")
    st.write("Segments shoppers using an augmented **RFM-S (Recency, Frequency, Monetary + Sentiment)** model.")

    seg_c1, seg_c2 = st.columns([1.2, 1])
    
    with seg_c1:
        st.markdown("##### 3D Feature Space: Recency, Monetary & Sentiment")
        fig_3d = px.scatter_3d(
            df_segments,
            x='recency', y='frequency', z='monetary',
            color='segment_name',
            hover_name='reviewer_name',
            hover_data=['country', 'avg_rating', 'avg_sentiment'],
            template="plotly_dark",
            color_discrete_sequence=['#38bdf8', '#a855f7', '#10b981', '#f59e0b']
        )
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            scene=dict(
                xaxis_title="Recency (Days)",
                yaxis_title="Frequency (# Purchases)",
                zaxis_title="Monetary Spend ($)"
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with seg_c2:
        st.markdown("##### Segment Distribution")
        seg_dist = df_segments['segment_name'].value_counts().reset_index()
        seg_dist.columns = ['Segment', 'Customers']
        fig_bar = px.bar(
            seg_dist, x='Customers', y='Segment',
            orientation='h', color='Segment',
            template="plotly_dark",
            color_discrete_sequence=['#38bdf8', '#a855f7', '#10b981', '#f59e0b']
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("##### Segment Behavioral Profile Comparison")
    segment_stats = df_segments.groupby('segment_name').agg(
        shopper_count=('user_id', 'count'),
        avg_spend=('monetary', 'mean'),
        avg_orders=('frequency', 'mean'),
        avg_recency_days=('recency', 'mean'),
        avg_sentiment_score=('avg_sentiment', 'mean'),
        avg_star_rating=('avg_rating', 'mean')
    ).reset_index()

    st.dataframe(
        segment_stats.style.format({
            "avg_spend": "${:,.2f}",
            "avg_orders": "{:.1f}",
            "avg_recency_days": "{:.0f} days",
            "avg_sentiment_score": "{:.3f}",
            "avg_star_rating": "{:.2f} ⭐"
        }),
        use_container_width=True
    )

    # Customer Lookup
    st.markdown("##### Inspect Individual Customer Persona")
    _nm2 = dict(zip(df_segments['user_id'], df_segments['reviewer_name']))
    selected_user = st.selectbox("Select Customer ID", sorted(df_segments['user_id'].tolist()), format_func=lambda u: u + "  —  " + _nm2.get(u, ""))
    u_row = df_segments[df_segments['user_id'] == selected_user].iloc[0]
    
    uc1, uc2, uc3, uc4 = st.columns(4)
    uc1.metric("Shopper Name", u_row['reviewer_name'])
    uc2.metric("Assigned Segment", u_row['segment_name'])
    uc3.metric("Lifetime Spend", f"${u_row['monetary']:,.2f}")
    uc4.metric("Avg Sentiment Score", f"{u_row['avg_sentiment']:.3f}")

# ----------------- MODULE 5: RECOMMENDATION ENGINE -----------------
elif app_mode == "Recommendation Engine":
    st.subheader("Sentiment-Aware Hybrid Recommendation Engine")
    st.write("Generates recommendations using **Item Content Similarity (TF-IDF)** and **Collaborative Filtering weighted by VADER Sentiment**.")

    rec_tab1, rec_tab2, rec_tab3 = st.tabs([
        "Personalized Shopper Recommendations",
        "Product Cross-Sell / Similarity",
        "Top-Trending Items"
    ])

    with rec_tab1:
        st.markdown("##### Customer-Specific Tailored Recommendations")
        _nm1 = dict(zip(df_segments['user_id'], df_segments['reviewer_name']))
        rec_user = st.selectbox("Select Shopper", sorted(df_segments['user_id'].tolist()), format_func=lambda u: u + "  —  " + _nm1.get(u, ""))
        
        # Display history
        u_history = df_reviews[df_reviews['user_id'] == rec_user]
        st.write(f"**Customer Purchase & Review History ({len(u_history)} items):**")
        st.dataframe(
            u_history[['product_name', 'category', 'rating', 'sentiment', 'review_title']].reset_index(drop=True),
            use_container_width=True
        )

        top_n = st.slider("Number of Recommendations", 2, 8, 4)
        recs = recommender.recommend_for_user(rec_user, top_n=top_n)

        st.markdown(f"##### Recommended Products for `{rec_user}`:")
        if len(recs) > 0:
            cols = st.columns(len(recs))
            for i, (_, prod) in enumerate(recs.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card" style="min-height: 180px;">
                        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600;">{prod['category']}</div>
                        <div style="font-weight:700; color:#f8fafc; font-size:1rem; margin: 6px 0;">{prod['product_name']}</div>
                        <div style="color:#38bdf8; font-weight:700; font-size:1.1rem;">${prod['unit_price']:.2f}</div>
                        <div style="color:#a78bfa; font-size:0.8rem; margin-top:4px;">Match Score: {prod.get('recommendation_score', 'Top Pick')}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No recommendations found for this shopper.")

    with rec_tab2:
        st.markdown("##### Similar Product Discovery (Content-Based via TF-IDF)")
        all_prods = recommender.product_catalog
        selected_prod_name = st.selectbox("Select Anchor Product", all_prods['product_name'].tolist())
        target_pid = all_prods[all_prods['product_name'] == selected_prod_name]['product_id'].iloc[0]

        sim_recs = recommender.recommend_similar_products(target_pid, top_n=4)
        
        st.markdown(f"##### Customers who viewed/bought `{selected_prod_name}` also liked:")
        if len(sim_recs) > 0:
            sim_cols = st.columns(len(sim_recs))
            for j, (_, sprod) in enumerate(sim_recs.iterrows()):
                with sim_cols[j]:
                    st.markdown(f"""
                    <div class="metric-card" style="min-height: 180px;">
                        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600;">{sprod['category']}</div>
                        <div style="font-weight:700; color:#f8fafc; font-size:1rem; margin: 6px 0;">{sprod['product_name']}</div>
                        <div style="color:#38bdf8; font-weight:700; font-size:1.1rem;">${sprod['unit_price']:.2f}</div>
                        <div style="color:#34d399; font-size:0.8rem; margin-top:4px;">Cosine Sim: {sprod['similarity_score']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No similar products found.")

    with rec_tab3:
        st.markdown("##### Highest Ranked Products by Sentiment & Popularity")
        trending = recommender.get_top_trending(top_n=8)
        st.dataframe(
            trending.style.format({
                "unit_price": "${:.2f}",
                "avg_rating": "{:.2f} ⭐"
            }),
            use_container_width=True
        )

# ----------------- MODULE 6: RAW DATA EXPLORER -----------------
elif app_mode == "Raw Data Explorer":
    st.subheader("Amazon Reviews Dataset Explorer")
    st.write(f"Displaying {len(filtered_df):,} enriched records conforming to the Kaggle Amazon Reviews schema.")
    
    st.dataframe(
        filtered_df[['review_id', 'user_id', 'reviewer_name', 'country', 'product_name', 'category', 'rating', 'vader_compound', 'sentiment', 'review_title', 'review_date']],
        use_container_width=True
    )

    csv_export = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Dataset as CSV",
        data=csv_export,
        file_name="amazon_reviews_processed.csv",
        mime="text/csv",
        type="primary"
    )
