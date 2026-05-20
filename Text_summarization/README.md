# 📰 Text Summarization — CNN/DailyMail Dataset (End-to-End NLP Project)

## 🔍 Overview

This project builds and compares two text summarization approaches — **Extractive (TF-IDF)** and **Abstractive (BART)** — on the CNN/DailyMail news dataset. It covers the full NLP pipeline from dataset loading and EDA to model inference and ROUGE-based evaluation.

The goal is to automatically condense long news articles into short, meaningful summaries and benchmark both approaches against human-written references.

---

## 📋 Project Workflow

### 1. Data Loading

* Dataset: **CNN/DailyMail 3.0.0** via HuggingFace Datasets (downloads automatically)
* 287,113 training | 13,368 validation | 11,490 test articles
* Worked with a **1,000-article sample** from the test split for speed

### 2. Exploratory Data Analysis (EDA)

* Analysed article and summary length distributions (words & sentences)
* Computed **compression ratio**: summaries average just **7.56%** of article length
* Median article length: **566 words** | Median summary length: **34 words**
* Visualised article vs summary length scatter and sentence count distribution

### 3. Extractive Summarization — TF-IDF Baseline

* Scored every sentence by TF-IDF importance
* Selected the **top 3 sentences** in original order — no new words generated
* Fast to run, no GPU required

### 4. Abstractive Summarization — BART

* Loaded **`facebook/bart-large-cnn`** via HuggingFace Transformers
* Generated fluent summaries in the model's own words, combining ideas across the article
* Ran on **GPU (T4)** for speed; truncated input to BART's 1,024-token limit

### 5. Evaluation with ROUGE

* Compared both approaches against human reference summaries on **100 articles**
* Metrics: **ROUGE-1** (word overlap), **ROUGE-2** (bigram overlap), **ROUGE-L** (longest common subsequence)

---

## 🤖 Approaches Compared

| Approach | Method | Speed |
|---|---|---|
| Extractive (TF-IDF) | Selects existing sentences by importance score | Fast — CPU only |
| Abstractive (BART) | Generates new sentences using a pretrained seq2seq model | Slower — GPU recommended |

---

## 📊 ROUGE Score Results

| Metric | Extractive (TF-IDF) | Abstractive (BART) |
|---|---|---|
| ROUGE-1 | 0.2486 | **0.3661** |
| ROUGE-2 | 0.0874 | **0.1644** |
| ROUGE-L | 0.1708 | **0.2840** |

BART outperforms the TF-IDF baseline across all three metrics.

---

## 🧠 Key Insights

* **BART generates far more concise summaries** — averaging ~30 words vs ~87 words for extractive, much closer to the human reference average of ~34 words.
* **Abstractive summaries read more naturally** — BART rephrases and combines ideas rather than copying raw sentences from the article.
* **Extractive is a strong baseline** for speed-constrained use cases — no model loading, no GPU required.
* **ROUGE-2 is the hardest metric** for both approaches, reflecting how difficult it is to match exact two-word phrases used by human writers.
* **7.56% compression ratio** in the CNN/DailyMail dataset makes this one of the most aggressive summarization tasks in NLP benchmarks.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Dataset:** CNN/DailyMail 3.0.0 (HuggingFace Datasets)
* **Pretrained Model:** `facebook/bart-large-cnn` (HuggingFace Transformers)
* **Extractive NLP:** NLTK, scikit-learn (TF-IDF)
* **Evaluation:** `rouge_score`
* **Visualization:** Matplotlib, Seaborn
* **Runtime:** Google Colab (T4 GPU)
