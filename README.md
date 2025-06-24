# Amazon Reviews Sentiment Analysis
This project analyzes Amazon product reviews using two sentiment analysis approaches: VADER (rule-based) and RoBERTa (transformer-based). It compares both models to evaluate sentiment accuracy across different star ratings.

## Features
- Preprocessing & EDA on 500 Amazon reviews

- Visualization of sentiment distributions across ratings

- VADER sentiment scoring using NLTK

- RoBERTa deep learning model for context-aware sentiment

- Model Comparison through scores and example mismatches

## Tech Stack
Python

NLTK, Transformers (HuggingFace)

Matplotlib, Seaborn

Pandas, NumPy

Torch, TensorFlow

## Setup
Install dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn nltk transformers torch tensorflow
Load the dataset:

## python
Copy
Edit
df = pd.read_csv('path_to/Reviews.csv')
Run the script to process and visualize sentiment scores.

$$ Highlights
VADER works well for basic sentiment.

RoBERTa captures nuanced sentiment, including sarcasm or context mismatches.

Side-by-side analysis of star ratings vs actual sentiment scores.
