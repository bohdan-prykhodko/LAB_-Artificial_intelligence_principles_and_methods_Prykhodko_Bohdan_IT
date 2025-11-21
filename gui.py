import os
import re
import threading

import customtkinter as ctk
from tkinter import messagebox

import torch
import joblib
import numpy as np

from googletrans import Translator
from transformers import BertTokenizer, BertForSequenceClassification

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class PhishingDetector(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Phishing Text Analyzer")
        self.geometry("900x700")

        self.translator = Translator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loading_label = ctk.CTkLabel(
            self,
            text="Loading ML Models...",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.loading_label.pack(pady=60)

        threading.Thread(target=self.load_resources, daemon=True).start()

    def load_resources(self):
        logreg_dir = r"D:\диплом\test_code_bert\env\logreg_model"
        cnn_dir = r"D:\диплом\test_code_bert\env\cnn_model"
        bert_dir = r"D:\диплом\test_code_bert\env\final_bert_phishing2"

        self.logreg_model = joblib.load(os.path.join(logreg_dir, "model.pkl"))
        self.logreg_tfidf = joblib.load(os.path.join(logreg_dir, "tfidf.pkl"))

        self.cnn_model = load_model(os.path.join(cnn_dir, "model.h5"))
        self.cnn_tokenizer = joblib.load(os.path.join(cnn_dir, "tokenizer.pkl"))
        self.cnn_max_len = 100

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_dir)
        self.bert_model.to(self.device)
        self.bert_model.eval()

        self.after(0, self.setup_ui)


    def setup_ui(self):
        self.loading_label.destroy()

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(
            top,
            text="Phishing Analyzer",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(side="left", padx=20)

        model_frame = ctk.CTkFrame(self, fg_color="transparent")
        model_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(
            model_frame,
            text="Select model:",
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(0, 8))

        self.model_selector = ctk.CTkComboBox(
            model_frame,
            values=["Logistic Regression", "Text CNN", "BERT Transformer"]
        )
        self.model_selector.set("BERT Transformer")
        self.model_selector.pack(side="left")

        ctk.CTkLabel(
            self,
            text="Enter text:",
            font=ctk.CTkFont(size=14)
        ).pack(anchor="w", padx=20, pady=(5, 0))

        self.input_textbox = ctk.CTkTextbox(
            self,
            height=150,
            wrap="word",
            font=ctk.CTkFont(size=13)
        )
        self.input_textbox.pack(fill="x", padx=20, pady=(0, 10))

        self.chunk_frame = ctk.CTkScrollableFrame(self)
        self.chunk_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self.analyze_btn = ctk.CTkButton(
            self,
            text="Analyze Text",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.validate_and_analyze
        )
        self.analyze_btn.pack(pady=5)

        self.result_label = ctk.CTkLabel(
            self,
            text="Waiting...",
            font=ctk.CTkFont(size=16)
        )
        self.result_label.pack(pady=5)

    def preprocess(self, text: str) -> str:
        t = re.sub(r"http[s]?://\S+|www\.\S+", "[link]", text)
        t = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;", " ", t)
        t = re.sub(r"[\W_]+", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def split_into_chunks(self, text: str, max_len: int = 512, stride: int = 50):
        ids = self.bert_tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return [text]
        chunks = []
        step = max_len - stride
        cls_id = self.bert_tokenizer.cls_token_id
        sep_id = self.bert_tokenizer.sep_token_id
        for i in range(0, len(ids), step):
            slice_ids = ids[i:i + max_len - 2]
            chunk_ids = [cls_id] + slice_ids + [sep_id]
            chunk_text = self.bert_tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks or [text]

    def show_chunks(self, chunks):
        for w in self.chunk_frame.winfo_children():
            w.destroy()
        for idx, chunk in enumerate(chunks, 1):
            fr = ctk.CTkFrame(self.chunk_frame, corner_radius=5)
            fr.pack(fill="x", pady=4, padx=4)
            ctk.CTkLabel(
                fr,
                text=f"Chunk {idx}",
                font=ctk.CTkFont(size=13, weight="bold")
            ).pack(anchor="w", padx=8, pady=(4, 0))
            ctk.CTkLabel(
                fr,
                text=chunk,
                wraplength=820,
                justify="left",
                font=ctk.CTkFont(size=12)
            ).pack(fill="x", padx=8, pady=(0, 6))

    def clear_chunks(self):
        for w in self.chunk_frame.winfo_children():
            w.destroy()

    def validate_and_analyze(self):
        text = self.input_textbox.get("0.0", "end").strip()
        if not text:
            messagebox.showerror("Error", "Text is empty.")
            return
        self.result_label.configure(text="Analyzing...", text_color="grey")
        self.analyze_btn.configure(state="disabled")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        raw_text = self.input_textbox.get("0.0", "end").strip()
        model_name = self.model_selector.get()

        if model_name == "Logistic Regression":
            processed = self.preprocess(raw_text)
            vec = self.logreg_tfidf.transform([processed])
            pred = self.logreg_model.predict(vec)[0]
            if hasattr(self.logreg_model, "predict_proba"):
                prob = self.logreg_model.predict_proba(vec)[0][pred] * 100.0
            else:
                prob = 0.0
            label = "Phishing" if pred == 1 else "Legitimate"
            self.after(0, lambda: self.finish(label, prob, None))
            return

        processed = self.preprocess(raw_text)
        try:
            lang = self.translator.detect(processed).lang
        except Exception:
            lang = "en"
        if lang == "uk":
            try:
                processed = self.translator.translate(processed, src="uk", dest="en").text
            except Exception:
                pass

        chunks = self.split_into_chunks(processed)

        if model_name == "BERT Transformer":
            label, conf = self.analyze_with_bert(chunks)
        elif model_name == "Text CNN":
            label, conf = self.analyze_with_cnn(chunks)
        else:
            label, conf = "Legitimate", 0.0

        self.after(0, lambda: self.finish(label, conf, chunks))

    def analyze_with_bert(self, chunks):
        total = torch.zeros(self.bert_model.config.num_labels, device=self.device)
        for ch in chunks:
            inp = self.bert_tokenizer(
                ch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                out = self.bert_model(**inp)
            probs = torch.nn.functional.softmax(out.logits, dim=1)[0]
            total += probs
        avg = total / len(chunks)
        pred = torch.argmax(avg).item()
        label = "Phishing" if pred == 1 else "Legitimate"
        conf = float(avg[pred].item() * 100.0)
        return label, conf

    def analyze_with_cnn(self, chunks):
        probs = []
        for ch in chunks:
            seq = self.cnn_tokenizer.texts_to_sequences([ch])
            pad = pad_sequences(seq, maxlen=self.cnn_max_len, padding="post")
            p = self.cnn_model.predict(pad, verbose=0)[0][0]
            probs.append(p)
        if not probs:
            return "Legitimate", 0.0
        avg = float(sum(probs) / len(probs))
        pred = 1 if avg > 0.5 else 0
        label = "Phishing" if pred == 1 else "Legitimate"
        conf = avg * 100.0 if pred == 1 else (1.0 - avg) * 100.0
        return label, conf

    def finish(self, label, conf, chunks):
        if chunks is not None:
            self.show_chunks(chunks)
        else:
            self.clear_chunks()
        color = "#FF3333" if label == "Phishing" else "#33FF77"
        self.result_label.configure(
            text=f"{label} detected ({conf:.2f}% confidence)",
            text_color=color
        )
        self.analyze_btn.configure(state="normal")
        

if __name__ == "__main__":
    app = PhishingDetector()
    app.mainloop()
