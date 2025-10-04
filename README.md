# Sentiment Analysis with TF-IDF and Logistic Regression

This repository contains a Jupyter notebook demonstrating sentiment analysis on customer reviews using TF-IDF vectorization and Logistic Regression.

## Contents
- `notebook.ipynb` - Jupyter notebook with preprocessing, modeling, evaluation, and saving of the model.
- `reviews.csv` - small sample dataset with `review` and `sentiment` columns.
- `requirements.txt` - Python dependencies.

## How to run
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```
2. Start Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Replace `reviews.csv` with your own dataset if desired (columns: `review`, `sentiment`).

## Notes
- The notebook demonstrates:
  - Text preprocessing (lowercasing, punctuation removal, stopword removal).
  - TF-IDF vectorization.
  - Logistic Regression model training with GridSearchCV.
  - Evaluation (accuracy, classification report, confusion matrix).
  - Saving the trained model with `joblib`.
