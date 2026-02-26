# Spam Message Detection (TF-IDF + Naive Bayes)

## 1) Project Title
**Spam Message Detection**

## 2) Problem Statement
The goal of this project is to automatically classify text messages into two categories:
- **ham**: normal, non-spam message
- **spam**: unwanted promotional or fraudulent message

This is a classic **text classification** problem. We solve it with:
- **TF-IDF (Term Frequency - Inverse Document Frequency)** for converting text into numbers
- **Multinomial Naive Bayes** for fast and effective spam classification

## 3) Dataset Description
This project uses a small custom CSV dataset located at:

`data/spam_sample.csv`

It contains two columns:
- `label`: target class (`ham` or `spam`)
- `message`: raw text message

The dataset is intentionally small and beginner-friendly so the full workflow is easy to understand.

## 4) Full Python Code
The complete runnable code is in:

`spam_message_detection.py`

It includes:
- data loading
- preprocessing with TF-IDF
- model training with Naive Bayes
- evaluation metrics
- visualization plots
- sample predictions

## 5) Explanation

### A) Data Loading
- We use `pandas.read_csv()` to load the CSV file into a DataFrame.
- We print the first rows, shape, and class distribution to inspect the data.

### B) Preprocessing
- Split the dataset into train/test sets using `train_test_split`.
- Use `TfidfVectorizer` to transform text into numerical feature vectors.
- Fit TF-IDF on training data only, then transform test data.

### C) Model Training
- Train a `MultinomialNB` model on TF-IDF features.
- This model is simple, fast, and effective for bag-of-words style text data.

### D) Evaluation
- Predict on test data.
- Compute:
  - Accuracy
  - Classification report (precision, recall, F1-score)
  - Confusion matrix

### E) Visualization
The script saves two plots:
- `class_distribution.png`: bar chart of ham vs spam samples
- `confusion_matrix.png`: model prediction quality by class

## 6) How to Run

### Requirements
- Python **3.10+**
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

### Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run the project
```bash
python spam_message_detection.py
```

### Expected output
- Console logs for:
  - dataset preview
  - model metrics
  - sample predictions
- Image files:
  - `class_distribution.png`
  - `confusion_matrix.png`
