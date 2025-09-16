# ==============================================
# Fake News Detection Project
# ==============================================

# Step 1: Import Libraries
import os
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# ==============================================
# Step 2: Load Dataset
# ==============================================
# Make sure fake.csv and true.csv are in the same folder
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Add labels
fake_df['label'] = 'fake'
true_df['label'] = 'real'

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# OUTPUT: First 5 rows of combined dataset
print(df.head())


# ==============================================
# Step 3: Text Cleaning
# ==============================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'<.*?>+', '', text)                  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)                     # remove newlines
    text = re.sub(r'\w*\d\w*', '', text)                # remove numbers
    text = " ".join([word for word in text.split() if word not in stop_words]) # remove stopwords
    return text

# Apply cleaning
df['cleaned_text'] = df['text'].apply(lambda x: clean_text(str(x)))

# OUTPUT: Show original and cleaned text
print(df[['text', 'cleaned_text']].head())


# ==============================================
# Step 4: Exploratory Data Analysis
# ==============================================

# 4.1 Distribution of real vs fake
sns.countplot(x='label', data=df, palette='viridis')
plt.title("Distribution of Real vs Fake News")
plt.show()   # OUTPUT: Bar chart

# 4.2 Word Cloud for Fake News
fake_text = " ".join(df[df['label'] == 'fake']['cleaned_text'])
fake_wc = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
plt.imshow(fake_wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud - Fake News")
plt.show()   # OUTPUT: Word cloud image

# 4.3 Word Cloud for Real News
real_text = " ".join(df[df['label'] == 'real']['cleaned_text'])
real_wc = WordCloud(width=800, height=400, background_color='white').generate(real_text)
plt.imshow(real_wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud - Real News")
plt.show()   # OUTPUT: Word cloud image


# ==============================================
# Step 5: TF-IDF Feature Extraction
# ==============================================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ==============================================
# Step 6: Model Training and Evaluation
# ==============================================

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))        # OUTPUT: Accuracy score
print(classification_report(y_test, y_pred_lr))              # OUTPUT: Precision/Recall/F1
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))  # OUTPUT: Matrix

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))        # OUTPUT: Accuracy score
print(classification_report(y_test, y_pred_nb))              # OUTPUT: Precision/Recall/F1
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))  # OUTPUT: Matrix

# Support Vector Machine
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\n--- Support Vector Machine ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))       # OUTPUT: Accuracy score
print(classification_report(y_test, y_pred_svm))             # OUTPUT: Precision/Recall/F1
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm)) # OUTPUT: Matrix

# Compare all models
results = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Naive Bayes": accuracy_score(y_test, y_pred_nb),
    "SVM": accuracy_score(y_test, y_pred_svm)
}
print("\nModel Comparison:", results)   # OUTPUT: Dictionary with accuracies


# ==============================================
# Step 7: Save Final Model (Logistic Regression chosen)
# ==============================================
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump((lr, tfidf), f)

print("\nFinal model saved as fake_news_model.pkl")   # OUTPUT: Confirmation


# ==============================================
# Step 8: Load Model and Predict New Samples
# ==============================================
with open("fake_news_model.pkl", "rb") as f:
    loaded_model, loaded_tfidf = pickle.load(f)

# Example prediction
sample_text = ["Breaking news: NASA confirms water on Mars!"]
sample_cleaned = [clean_text(sample_text[0])]
sample_tfidf = loaded_tfidf.transform(sample_cleaned)
prediction = loaded_model.predict(sample_tfidf)

print("\nPrediction for sample text:", prediction[0])   # OUTPUT: 'real' or 'fake'
