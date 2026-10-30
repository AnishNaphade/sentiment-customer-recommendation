# Sentiment Analysis, Customer Segmentation and Recommendation System

[![Publication](https://img.shields.io/badge/Publication-Scopus%20Elsevier%20ACT--2025-blue?style=for-the-badge)](https://grenze.org/)
[![Java](https://img.shields.io/badge/Java-11+-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Stanford CoreNLP](https://img.shields.io/badge/Stanford_CoreNLP-4.5+-red?style=for-the-badge)](https://stanfordnlp.github.io/CoreNLP/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

An integrated deep learning and machine learning framework combining **Java-based Stanford CoreNLP** sentiment analysis with **Python-based unsupervised customer segmentation (K-Means + KNN)** and **Cosine Similarity recommendation algorithms**, as published in **Grenze / Scopus ACT-2025**.

---

## 🔬 Research & Architecture Overview

As formulated in the research paper:
> *"This research proposes an integrated approach combining Java-based sentiment analysis using Stanford CoreNLP with Python-based customer segmentation and recommendation techniques. By vectorizing customer data and applying clustering techniques, the system generates precise, context-aware product recommendations."*

```
                 Figure 1: End-to-End Processing Flow
  +--------------+       +-----------------+       +--------------------+
  | Collect Data |  ==>  | Preprocess Data |  ==>  | Sentiment Analysis |
  | (E-Commerce) |       | (Normalization) |       | (Stanford CoreNLP) |
  +--------------+       +-----------------+       +--------------------+
                                                             ||
                                                             \/
  +--------------+       +-----------------+       +--------------------+
  | Develop Recs |  <==  | Generate Insights | <== |  Aggregate Results |
  | (Cosine Sim) |       | (K-Means + KNN) |       |  (Feature Vectors) |
  +--------------+       +-----------------+       +--------------------+
```

---

## 🌟 Core System Components

### 1. Java-Based Stanford CoreNLP Sentiment Framework (`src/java/...`)
- Implements linguistic dependency parsing, Part-of-Speech (POS) tagging, and recursive deep sentiment trees.
- Extracts continuous emotion scores across 5 sentiment levels:
  - `0: Very Negative`, `1: Negative`, `2: Neutral`, `3: Positive`, `4: Very Positive`.
- Generates continuous sentiment intensity values representing customer emotional states beyond binary positive/negative labels.

### 2. Customer Segmentation Engine (`src/customer_segmentation.py`)
- **K-Means Clustering:** Groups customers into $k=3$ distinct behavioral personas based on transaction volume, monetary spend, and continuous sentiment profiles (optimized via the Elbow heuristic).
- **K-Nearest Neighbors (KNN):** Implements dynamic adaptation by mapping new incoming customers into appropriate clusters via $k$-NN classification ($k=5$).

### 3. Context-Aware Cosine Recommendation Engine (`src/recommender.py`)
- Employs pairwise **Cosine Similarity** between high-dimensional customer preference vectors and product feature matrices.
- Integrates sentiment polarity directly into recommendation weights, ensuring suggestions are emotionally and contextually relevant.

---

## 📂 Repository Structure

```
.
├── pom.xml                               # Maven build configuration for Java Stanford CoreNLP
├── src/
│   ├── java/
│   │   └── com/sentiment/recommendation/
│   │       └── CoreNLPSentimentAnalyzer.java # Java Stanford CoreNLP pipeline
│   ├── sentiment_analyzer.py             # Stanford CoreNLP Python bridge & linguistic analyzer
│   ├── customer_segmentation.py          # K-Means clustering + KNN classifier
│   ├── recommender.py                    # Cosine Similarity recommendation engine
│   └── app.py                            # Streamlit interactive dashboard
├── data/
│   ├── amazon_reviews.csv                # E-commerce review and transaction dataset
│   └── customer_segments.csv             # Clustered customer profiles
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Run the Python Application & Dashboard
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

### 2. Build and Run the Java Stanford CoreNLP Module
```bash
mvn clean compile
mvn exec:java -Dexec.mainClass="com.sentiment.recommendation.CoreNLPSentimentAnalyzer"
```

---

## 📜 Citation

If you find this research framework or codebase useful, please cite:

```bibtex
@article{kulkarni2025sentiment,
  title={Sentiment Analysis, Customer Segmentation and Recommendation System},
  author={Kulkarni, Milind and Naphade, Anish and Jagtap, Alankar and Ambilwade, Aniket and Wagh, Ashlesha and Tamboli, Arman},
  journal={Grenze International Journal of Engineering and Technology (GIJET)},
  volume={11},
  number={2},
  pages={11937--11944},
  year={2025},
  publisher={Grenze Scientific Society}
}
```

---

## 👨‍💻 Author
**Anish Naphade**  
B.Tech in Artificial Intelligence & Data Science, Vishwakarma Institute of Technology, Pune  
- [GitHub](https://github.com/AnishNaphade)
- [LinkedIn](https://linkedin.com/in/anishnaphade)
- Email: anishnaphade@gmail.com
