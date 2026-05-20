# Amazon Alexa Reviews — Sentiment Analysis

**A binary text classification project that predicts whether an Amazon Alexa product review is positive or negative using NLP and machine learning.**

## Executive Summary
- **Project Scope:** Built an end-to-end NLP pipeline to classify **3,150 verified Amazon Alexa reviews** as positive or negative.
- **Modelling:** Applied **TF-IDF vectorization** and **Logistic Regression** with class balancing to handle the imbalanced dataset (92% positive, 8% negative).
- **Outcomes:** Achieved **94.78% accuracy** with strong recall on the minority (negative) class — a reliable classifier for real-world review analysis.

**Notebook:**

[alexa_sentiment_analysis.ipynb](alexa_sentiment_analysis.ipynb) — Full pipeline: EDA, text cleaning, model training, evaluation, and live prediction.

## Tech Stack
- **Language:** Python
- **Core Libraries:** pandas, NumPy, scikit-learn, Matplotlib, seaborn, WordCloud
- **NLP:** TF-IDF Vectorization (top 5,000 features)
- **Model:** Logistic Regression (`class_weight='balanced'`)
- **Development:** Jupyter Notebook / VS Code

## Analysis Overview

### Pipeline Structure

**Step 1 — EDA**
Explored the class distribution (2,893 positive vs 257 negative reviews) and visualised word clouds to identify the most common language in each sentiment group.

**Step 2 — Text Cleaning**
Lowercased text, removed punctuation and numbers, and stripped empty reviews — reducing the dataset from 3,150 to 3,065 usable rows.

**Step 3 — Feature Engineering**
Converted cleaned reviews to a TF-IDF matrix (2,452 training reviews × 3,963 features) using an 80/20 stratified train-test split.

**Step 4 — Modelling & Evaluation**
Trained Logistic Regression on TF-IDF features. Evaluated with accuracy, precision, recall, F1-score, and a confusion matrix.

**Step 5 — Inference**
Built a `predict_review()` function that takes any raw review text and returns a positive/negative prediction with confidence score.

## Key Insights
- **94.78% overall accuracy** on the held-out test set.
- **Negative recall of 81%** — the model correctly flags most genuine negative reviews despite the class imbalance.
- Top positive signal words: *love*, *great*, *easy*, *perfect*, *amazing*.
- Top negative signal words: *waste*, *disappointed*, *return*, *terrible*, *stopped*.
- Neutral-sounding reviews (e.g. "It is okay, nothing special") are correctly classified as positive with lower confidence (~60%), showing the model is appropriately uncertain.
