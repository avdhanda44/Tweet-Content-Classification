# NLP-Tweet-Content-Classification

An end-to-end NLP portfolio project that classifies tweets as **hate speech**, **offensive language**, or **neither**.

## Problem statement

Online platforms receive large volumes of user-generated text that may contain hate speech, offensive language, or harmless content. Manual review is slow and inconsistent. This project investigates whether classical NLP and machine-learning techniques can classify tweets into these three categories and assist human moderators, while measuring errors, class imbalance, and responsible-use limitations.

## What this demonstrates

- Dataset inspection and class-balance analysis
- Social-media text cleaning for URLs, mentions, retweets, HTML entities, and punctuation
- A controlled comparison of Bag-of-Words and TF-IDF features
- Linear Logistic Regression and Linear SVM classifiers
- Stratified train/test evaluation
- Per-class precision, recall, F1-score, macro F1, and a confusion matrix
- Error analysis and a reusable `predict_tweet` inference function
- Guided NLP concepts lab with real NLTK tokenization, POS tagging, named entity recognition, stemming, lemmatization, and VADER sentiment

## Project structure

- `data/raw/labeled_data.csv`: original labeled dataset; never edit this file
- `data/processed/tweets_cleaned.csv`: generated model-ready data
- `tweet_classification.ipynb`: complete experiment and results narrative
- `data_dictionary.md`: raw dataset column definitions
- `src/preprocess.py`: reproducible raw-to-processed data pipeline
- `src/train.py`: reproducible model training and metric export
- `src/visualize_data.py`: exploratory visualization pipeline
- `reports/figures/`: generated charts for the project report
- `reports/model_results.csv`: saved model comparison metrics
- `tests/`: preprocessing tests
- `models/`: saved model artifacts
- `requirements.txt`: Python dependencies

## How to read the result

Macro F1 is the primary comparison metric because it gives equal importance to all three classes. The confusion matrix and error examples are essential: hate speech and offensive language are related but distinct labels, and disagreements can reflect ambiguity in the annotations rather than a simple modeling failure.

## Responsible use

This is a research and portfolio baseline, not an autonomous moderation tool. The data may contain annotation bias, offensive language, identity-related terms, sarcasm, and missing context. Any real deployment would need human review, subgroup fairness checks, threshold calibration, privacy safeguards, and monitoring for distribution shift.

## Running it

Install dependencies from `requirements.txt`, then run `python3 src/preprocess.py`, `python3 src/visualize_data.py`, and `python3 src/train.py`. Open `tweet_classification.ipynb` in VS Code or Jupyter and run the cells from top to bottom. Run `pytest` for the preprocessing test. Every transformation writes to `data/processed`, `reports`, or `models`; `data/raw` remains unchanged.
