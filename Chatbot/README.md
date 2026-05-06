# NLP Chatbot Question Answering System

## Project Overview

This project builds a deep learning-based Question Answering (QA) chatbot using Natural Language Processing (NLP) and TensorFlow/Keras.

The chatbot is trained to understand short stories and answer related questions using a memory network architecture with LSTM layers and word embeddings.

The project demonstrates practical applications of:

* Natural Language Processing (NLP)
* Deep Learning
* Sequence Modeling
* Neural Networks
* TensorFlow/Keras

---

# Objective

The main goal of this project is to create a chatbot capable of:

* Reading textual story data
* Understanding context
* Processing questions
* Predicting accurate answers

The model learns relationships between stories and questions to generate relevant responses.

---

# Dataset

The project uses preprocessed QA datasets stored as:

* `train_qa.txt`
* `test_qa.txt`

The data consists of:

* Story text
* Related questions
* Correct answers

The datasets are loaded using Python's `pickle` module.

---

# Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Google Colab

---

# Project Workflow

## 1. Data Loading

The training and testing datasets are loaded from Google Drive using Python.

```python
with open("train_qa.txt", 'rb') as f:
    train_data = pickle.load(f)
```

---

## 2. Vocabulary Creation

A vocabulary set is generated from:

* Stories
* Questions
* Answers

Additional labels such as:

* `yes`
* `no`

are also added to improve response handling.

---

## 3. Tokenization & Sequence Processing

Text data is converted into numerical sequences using Keras Tokenizer.

Key preprocessing steps include:

* Tokenization
* Padding sequences
* Vectorization
* Sequence encoding

This allows neural networks to process textual information efficiently.

---

## 4. Neural Network Architecture

The chatbot model uses:

* Embedding layers
* Dropout layers
* LSTM layers
* Memory-based attention mechanism

The architecture is built using TensorFlow/Keras Functional API.

Main components:

* Input Encoder
* Question Encoder
* Attention Mechanism
* LSTM Answer Generator

---

# Model Training

The model is trained using:

```python
model.fit()
```

Training configuration:

* Optimizer: RMSprop
* Loss Function: Categorical Crossentropy
* Metric: Accuracy
* Epochs: 100

---

# Performance Visualization

Training and validation accuracy are visualized using Matplotlib.

Graphs help evaluate:

* Learning progress
* Model performance
* Overfitting/underfitting trends

---

# Model Saving

The trained chatbot model is saved using:

```python
model.save('chatbot.keras')
```

This allows the trained model to be reused without retraining.

---

# Key Features

* NLP-based chatbot system
* Story comprehension and QA prediction
* Sequence vectorization
* Deep learning memory network
* LSTM-based response generation
* TensorFlow/Keras implementation

---
