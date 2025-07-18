import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords
nltk.download('stopwords')
print("Stopwords ready, continuing with the script...")
from nltk.corpus import stopwords

# ---------------------------
# Step 1: Load and Combine Data
# ---------------------------
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0
real['label'] = 1

data = pd.concat([fake, real], axis=0)
data = data.sample(n=5000).reset_index(drop=True)  # Use 5k rows instead of all ~44k - Shuffle

# ---------------------------
# Step 2: Preprocessing
# ---------------------------

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(clean_text)

# ---------------------------
# Step 3: Vectorization
# ---------------------------

vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

# ---------------------------
# Step 4: Train/Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Step 5: Model Training
# ---------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Step 6: Evaluation
# ---------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ---------------------------
# Step 7: Save Model and Vectorizer
# ---------------------------

joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved to disk.")
