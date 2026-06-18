# Day 4 — NLP + Generative AI Text Analysis
# Kaggle 5-Day Vibe Coding Program
# Author: Aman Trivedi
# Using AI tools to analyze and generate text — this was super fun!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import re
import warnings; warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

print("📝 Day 4 — NLP + Generative AI Text Analysis")
print("=" * 55)

texts = [
    "Machine learning is transforming the way we build intelligent systems and solve complex problems.",
    "Deep learning neural networks can recognize patterns in images with superhuman accuracy.",
    "Natural language processing enables computers to understand and generate human language.",
    "Quantum computing uses quantum mechanics to solve problems impossible for classical computers.",
    "Data science combines statistics, programming and domain knowledge to extract insights from data.",
    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
    "Computer vision algorithms can detect objects, faces and scenes in real-time video streams.",
    "Transfer learning allows models trained on large datasets to be fine-tuned for specific tasks.",
    "The transformer architecture revolutionized NLP and gave birth to models like GPT and BERT.",
    "Generative AI can create realistic images, music, code and text from simple prompts.",
    "Feature engineering is the process of creating meaningful input variables for machine learning.",
    "Cross-validation is a robust technique to evaluate model performance on unseen data.",
    "Overfitting occurs when a model learns noise in training data and fails to generalize.",
    "Ensemble methods combine multiple models to achieve better predictive performance.",
    "Explainable AI helps users understand how black-box models make their predictions.",
]

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 3]
    return ' '.join(tokens)

processed = [preprocess(t) for t in texts]

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
sentiments = [sia.polarity_scores(t) for t in texts]
compounds = [s['compound'] for s in sentiments]
labels = ['Positive' if c > 0.05 else 'Negative' if c < -0.05 else 'Neutral' for c in compounds]
print("\nSentiment Analysis Results:")
for i, (t, l, c) in enumerate(zip(texts[:5], labels[:5], compounds[:5])):
    print(f"  [{l:8s} {c:+.3f}] {t[:60]}...")

# TF-IDF Keywords
tfidf = TfidfVectorizer(max_features=20, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(processed)
keywords = tfidf.get_feature_names_out()
scores = tfidf_matrix.toarray().mean(axis=0)
kw_df = pd.DataFrame({'keyword': keywords, 'score': scores}).sort_values('score', ascending=False)

# Topic Modeling with LDA
cv = CountVectorizer(max_features=30)
cv_matrix = cv.fit_transform(processed)
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(cv_matrix)
feature_names = cv.get_feature_names_out()
print("\n📚 Top Topics (LDA):")
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
    print(f"  Topic {idx+1}: {', '.join(top_words)}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Day 4 — NLP & Text Analysis', fontsize=14, fontweight='bold', color='#1B3A6B')

colors = ['#2ecc71' if l == 'Positive' else '#e74c3c' if l == 'Negative' else '#f39c12' for l in labels]
axes[0].bar(range(len(compounds)), compounds, color=colors, alpha=0.85, edgecolor='white')
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Sentiment Scores'); axes[0].set_xlabel('Text Index'); axes[0].set_ylabel('Compound Score')
handles = [mpatches.Patch(color='#2ecc71', label='Positive'), mpatches.Patch(color='#e74c3c', label='Negative'), mpatches.Patch(color='#f39c12', label='Neutral')]
axes[0].legend(handles=handles, fontsize=8)

kw_top = kw_df.head(10)
axes[1].barh(kw_top['keyword'], kw_top['score'], color='#1B3A6B', alpha=0.85)
axes[1].set_title('Top TF-IDF Keywords'); axes[1].set_xlabel('Avg TF-IDF Score')

topic_dist = lda.transform(cv_matrix)
for i in range(3):
    axes[2].bar(range(len(texts)), topic_dist[:,i], label=f'Topic {i+1}', alpha=0.7, bottom=topic_dist[:,:i].sum(axis=1) if i > 0 else None)
axes[2].set_title('Topic Distribution per Text'); axes[2].set_xlabel('Text Index'); axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('day4_nlp_analysis.png', dpi=150)
plt.show()
print("\n✅ Day 4 Complete!")
