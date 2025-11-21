import os
import pandas as pd
import numpy as np
import json
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("phishing_dataset_bert.csv")

train_val_df, test_df = train_test_split(
    df, test_size=0.10, stratify=df["label"], random_state=42
)

train_df, val_df = train_test_split(
    train_val_df, test_size=0.10, stratify=train_val_df["label"], random_state=42
)

X_train, y_train = train_df["body"], train_df["label"]
X_val, y_val = val_df["body"], val_df["label"]
X_test, y_test = test_df["body"], test_df["label"]

tokenizer = Tokenizer(num_words=20000, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

model = Sequential([
    Embedding(20000, 128, input_length=max_len),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_pad, y_train,
    epochs=12,
    batch_size=64,
    validation_data=(X_val_pad, y_val),
    callbacks=[early_stop]
)

y_pred_prob = model.predict(X_test_pad).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
)
acc = accuracy_score(y_test, y_pred)

print("Text CNN Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

save_dir = "cnn_model"
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, "model.h5"))
joblib.dump(tokenizer, os.path.join(save_dir, "tokenizer.pkl"))

metrics = {
    "model": "Text CNN",
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
