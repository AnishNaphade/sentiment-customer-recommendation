# Sentiment Analysis, Customer Segmentation & Recommendation System

An end-to-end Machine Learning and Natural Language Processing (NLP) web platform built using the Amazon Reviews Dataset format.

---

## 🌟 Key Features
1. **Sentiment Analysis Engine**:
   - VADER Lexicon + TextBlob integration.
   - Computes compound sentiment, polarity, and subjectivity.
   - Real-time live review sentiment diagnostic tool.

2. **Customer Segmentation (RFM-S + K-Means)**:
   - Augments traditional Recency, Frequency, and Monetary (RFM) modeling with **Sentiment Health**.
   - Identifies business personas (*Champions & Brand Advocates*, *High-Value At-Risk Buyers*, *Hibernating Shoppers*, etc.).
   - Interactive 3D feature space visualization.

3. **Hybrid Recommendation System**:
   - **Content-Based Filtering**: TF-IDF cosine similarity over product metadata and review lexicons.
   - **Sentiment-Weighted Collaborative Filtering**: User-Item interaction matrix boosted by positive review scores.
   - Top-trending popularity ranker.

4. **Streamlit Executive Web Application**:
   - Polished glassmorphism dark-mode UI.
   - Interactive Plotly figures, filters, and real-time inference.

---

## 🚀 Quickstart

### 1. Run the Streamlit Application
```powershell
cd C:\PROJECTS\SENTIMENT
streamlit run src\app.py
```

### 2. Project Directory Structure
```
C:\PROJECTS\SENTIMENT\
├── data/
│   ├── amazon_reviews.csv              # Full reviews & transaction dataset
│   ├── reviews_with_sentiment.csv      # NLP-scored dataset
│   └── customer_segments.csv          # Segmented customer profiles
├── src/
│   ├── __init__.py
│   ├── sentiment_analyzer.py          # VADER + TextBlob sentiment logic
│   ├── customer_segmentation.py       # RFM-S + K-Means clustering
│   ├── recommender.py                 # Hybrid TF-IDF & Collaborative recommender
│   └── app.py                         # Interactive Streamlit dashboard
├── models/
│   ├── kmeans_model.joblib            # Trained K-Means model
│   ├── scaler.joblib                  # Standard scaler
│   └── labels_map.joblib              # Cluster persona mappings
└── README.md
```
