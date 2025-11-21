import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_dataset(path):
    df = pd.read_csv(path)

    train_val_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=42
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=0.10,
        stratify=train_val_df["label"], random_state=42
    )

    return train_df, val_df, test_df


def tokenize_datasets(train_df, val_df, test_df):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def encode(batch):
        tokens = tokenizer(
            batch["body"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        tokens["labels"] = batch["label"]
        return tokens

    train_ds = Dataset.from_pandas(train_df).map(encode, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(encode, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(encode, batched=True)

    for name in ["train_ds", "val_ds", "test_ds"]:
        ds = locals()[name]
        remove_cols = [c for c in ds.column_names if c not in ["input_ids", "attention_mask", "labels"]]
        ds = ds.remove_columns(remove_cols)
        ds.set_format("torch")
        locals()[name] = ds

    return train_ds, val_ds, test_ds, tokenizer


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="binary"
    )
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_model(train_ds, val_ds, tokenizer):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="bert_phishing",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,

        logging_steps=200,

        learning_rate=2e-5,
        weight_decay=0.01,

        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    return trainer


def evaluate_and_save(trainer, test_ds, tokenizer):
    print("Evaluating model...")
    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)

    save_dir = "final_bert_phishing3"
    print(f"Saving model to: {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    return save_dir


def test_predictions(model_path):
    clf = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        return_all_scores=False
    )

    samples = [
        "Your account has been compromised! Click here to reset your password.",
        "Дякуємо за покупку на нашому сайті – ваш заказ успішно оформлено."
    ]
    print("Example prediction:")
    print(clf(samples))


if __name__ == "__main__":
    dataset_path = r"D:\диплом\datasets\phishing_dataset_bert.csv"

    print("Loading dataset...")
    train_df, val_df, test_df = load_dataset(dataset_path)

    print("Tokenizing...")
    train_ds, val_ds, test_ds, tokenizer = tokenize_datasets(train_df, val_df, test_df)

    print("Training model...")
    trainer = train_model(train_ds, val_ds, tokenizer)

    save_dir = evaluate_and_save(trainer, test_ds, tokenizer)

    print("Testing predictions...")
    test_predictions(save_dir)

    print("Done.")
