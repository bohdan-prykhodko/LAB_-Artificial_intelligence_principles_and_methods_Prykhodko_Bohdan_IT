import os
import pandas as pd
import numpy as np
import json
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv("phishing_dataset_bert.csv")

train_val_df, test_df = train_test_split(
    df, test_size=0.10, stratify=df["label"], random_state=42
)

train_df, val_df = train_test_split(
    train_val_df, test_size=0.10, stratify=train_val_df["label"], random_state=42
)

X_train, y_train = train_df["body"], train_df["label"]
X_test, y_test = test_df["body"], test_df["label"]

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
)
acc = accuracy_score(y_test, y_pred)

print("Logistic Regression Baseline:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

save_dir = "logreg_model"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "model.pkl"))
joblib.dump(tfidf, os.path.join(save_dir, "tfidf.pkl"))

metrics = {
    "model": "Logistic Regression",
    "accuracy": float(acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "dataset": "phishing_dataset_bert.csv",
    "test_size": 0.10,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(save_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
