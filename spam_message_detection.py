"""
Spam Message Detection Project
-----------------------------
A beginner-friendly machine learning project for classifying SMS-like text
messages as "spam" or "ham" (not spam).

Techniques used:
- TF-IDF vectorization for text preprocessing
- Multinomial Naive Bayes for classification

Libraries used:
- pandas
- numpy
- matplotlib
- scikit-learn
"""

# Standard library imports
from pathlib import Path

# Third-party imports (allowed beginner libraries)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# 1) DATA LOADING
# -----------------------------
# Build a path to the dataset file so the script works from the project root.
DATASET_PATH = Path("data/spam_sample.csv")

# Load the CSV file into a pandas DataFrame.
# Expected columns:
# - label: "ham" or "spam"
# - message: text content of the message
messages_df = pd.read_csv(DATASET_PATH)

print("First 5 rows of the dataset:")
print(messages_df.head(), "\n")

print("Dataset shape (rows, columns):", messages_df.shape)
print("Class distribution:")
print(messages_df["label"].value_counts(), "\n")


# -----------------------------
# 2) PREPROCESSING
# -----------------------------
# Separate features (X) and target labels (y).
X_text = messages_df["message"]
y = messages_df["label"]

# Split data into training and testing sets.
# - test_size=0.3 means 30% test data
# - random_state=42 ensures reproducible results
# - stratify=y keeps class balance similar in train/test sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# Convert text into numerical features using TF-IDF.
# TF-IDF gives higher importance to words that are unique and informative.
# - lowercase=True ensures case-insensitive processing
# - stop_words="english" removes common English filler words
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")

# Fit on training text and transform both train and test text.
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print("TF-IDF training matrix shape:", X_train_tfidf.shape)
print("TF-IDF testing matrix shape:", X_test_tfidf.shape, "\n")


# -----------------------------
# 3) MODEL TRAINING
# -----------------------------
# MultinomialNB works very well for word-count / TF-IDF style text features.
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# -----------------------------
# 4) EVALUATION
# -----------------------------
# Predict labels for test data.
y_pred = model.predict(X_test_tfidf)

# Compute accuracy.
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}\n")

# Show detailed precision/recall/F1 per class.
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix to see correct/incorrect class predictions.
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print("Confusion Matrix (rows=true, cols=pred):")
print(cm, "\n")


# -----------------------------
# 5) VISUALIZATION
# -----------------------------
# A) Bar chart of class counts
plt.figure(figsize=(7, 4))
messages_df["label"].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
plt.title("Class Distribution in Dataset")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()

# B) Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("Saved plot: class_distribution.png")
print("Saved plot: confusion_matrix.png\n")


# -----------------------------
# 6) SMALL DEMO PREDICTIONS
# -----------------------------
# Demonstrate predictions on new messages.
sample_messages = [
    "Congratulations! You won a free iPhone. Click to claim now!",
    "Hi, can you share the meeting agenda for tomorrow?",
    "Earn money fast from home with no experience required.",
]

sample_features = vectorizer.transform(sample_messages)
sample_predictions = model.predict(sample_features)

print("Demo predictions:")
for text, pred in zip(sample_messages, sample_predictions):
    print(f"- Message: {text}\n  Predicted label: {pred}\n")

# Print top words strongly associated with spam according to model log probabilities.
feature_names = np.array(vectorizer.get_feature_names_out())
spam_class_index = np.where(model.classes_ == "spam")[0][0]
spam_feature_scores = model.feature_log_prob_[spam_class_index]

top_indices = np.argsort(spam_feature_scores)[-10:][::-1]
print("Top words associated with spam class:")
for word in feature_names[top_indices]:
    print("-", word)
