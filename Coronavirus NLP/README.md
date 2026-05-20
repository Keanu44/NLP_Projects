# Corona NLP — Tweet Sentiment Classification

**A 5-class NLP project that classifies COVID-19 tweets by sentiment using TF-IDF and Logistic Regression.**

## Executive Summary
- **Project Scope:** Built a multi-class text classifier trained on **41,157 COVID-19 tweets** to predict one of five sentiment labels: Extremely Negative, Negative, Neutral, Positive, and Extremely Positive.
- **Modelling:** Used **TF-IDF with bigrams** (10,000 features) and **Logistic Regression** with class balancing to handle uneven sentiment distribution across the dataset.
- **Outcomes:** Achieved **54.19% accuracy** on a held-out test set of 3,798 tweets — strong for a 5-class problem where adjacent classes (e.g. Negative vs Extremely Negative) are intentionally difficult to distinguish.

**Notebook:**

[corona_nlp_classification.ipynb](corona_nlp_classification.ipynb) — Full pipeline: EDA, tweet cleaning, TF-IDF modelling, multi-class evaluation, and live prediction.

## Tech Stack
- **Language:** Python
- **Core Libraries:** pandas, NumPy, scikit-learn, Matplotlib, seaborn, WordCloud
- **NLP:** TF-IDF Vectorization with unigrams + bigrams (`ngram_range=(1, 2)`, top 10,000 features)
- **Model:** Logistic Regression (`class_weight='balanced'`, `max_iter=1000`)
- **Development:** Jupyter Notebook / VS Code

## Analysis Overview

### Dataset
| Split    | Rows   |
|----------|--------|
| Training | 41,157 |
| Test     | 3,798  |

**Sentiment classes (5):** Extremely Negative · Negative · Neutral · Positive · Extremely Positive

### Pipeline Structure

**Step 1 — EDA**
Analysed sentiment distribution (Positive is the largest class at 11,422 tweets), tweet length distribution (median ~152 characters), and word clouds for positive vs negative tweet clusters.

**Step 2 — Text Cleaning**
Removed URLs, @mentions, #hashtags, punctuation, and numbers — leaving only lowercase alphabetic content for the model to learn from.

**Step 3 — Feature Engineering**
Mapped 5 sentiment labels to integers (0–4) and vectorised tweets with TF-IDF, capturing both individual words and two-word phrases relevant to COVID-19 discourse (e.g. "social distancing", "panic buying").

**Step 4 — Modelling & Evaluation**
Trained Logistic Regression and evaluated with per-class precision, recall, F1-score, a confusion matrix, and a side-by-side actual vs predicted count chart.

**Step 5 — Inference**
Built a `predict_tweet()` function that classifies any raw tweet text into one of the five sentiment categories with a confidence score.

## Key Insights
- **54.19% accuracy** on 5-class classification — a reasonable baseline given the subtle boundary between adjacent sentiment classes.
- **Neutral (63% F1) and Extremely Positive (64% F1)** are the best-performing classes; the model struggles most with mid-range sentiments (Negative and Positive, ~44–45% F1).
- Bigrams significantly improve signal — phrases like *panic buying* and *stay home* carry stronger sentiment cues than individual words alone.
- Top predictive words: *thank*, *grateful*, *together* (positive); *panic*, *disaster*, *lockdown* (negative); *government*, *announced*, *restrictions* (neutral).
- The model correctly identifies extreme sentiments (job loss, vaccine optimism) with higher confidence than ambiguous mid-range tweets.
