#  Fake News Classifier (NLP + ML)

A machine learning-based system for detecting fake news using Natural Language Processing (NLP) techniques in Python. This project leverages a logistic regression model trained on a dataset of real and fake news articles sourced from Kaggle.

##  Project Overview

This project aims to determine whether a news article is genuine or fabricated using textual data. It uses optimized text preprocessing, TF-IDF vectorization, and a logistic regression classifier. The model achieved **96.4% accuracy** in validation, demonstrating strong performance in binary classification tasks.

---

##  Dataset

The dataset was obtained from [Kaggle's Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It contains labeled news articles with the following structure:

- `Fake.csv` — contains fabricated news stories (`label = 0`)
- `True.csv` — contains legitimate news articles (`label = 1`)

Total size: ~44,000 articles (sampled to 5,000 for rapid experimentation)

---

##  Technologies Used

- **Python** (v3.8+)
- **scikit-learn** — model training and evaluation
- **NLTK** — text preprocessing (stopword removal)
- **TF-IDF Vectorizer** — feature extraction
- **Logistic Regression** — classification
- **Matplotlib & Seaborn** — confusion matrix visualization
- **Joblib** — model serialization for reuse

---

##  Features

- Clean and preprocess text using `NLTK` (lowercasing, punctuation removal, stopword filtering)
- Vectorize articles using `TfidfVectorizer`
- Train/test split with 80-20 ratio using `train_test_split`
- Logistic regression model trained on TF-IDF vectors
- Evaluation using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Heatmap visualization of the confusion matrix
- Export trained model and vectorizer using `joblib` for future inference

---

##  Model Performance

- **Accuracy:** 96.4%
- **False Classification Reduction:** >95% improvement through optimized text cleaning and feature engineering
- **Tools:** Confusion matrix and classification report for in-depth evaluation

---

##  Output Files

After training, the following files are saved:

- `fake_news_model.pkl` — Trained logistic regression model
- `tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer

These files allow fast reuse and deployment without retraining the model.

---

##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python fake_news_classifier.py
```

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this software in both personal and commercial projects.

## Author

Created by Henil Daslaniya. Contributions and suggestions are welcome!
