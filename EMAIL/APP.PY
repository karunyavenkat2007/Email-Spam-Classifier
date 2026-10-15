"""
Email Spam Classifier
=====================
Trains Naive Bayes, Logistic Regression and SVM on the UCI SMS Spam Collection
(Kaggle: uciml/sms-spam-collection-dataset), compares them, visualizes the
top spam keywords, saves the best model and classifies new messages in real time.

Setup:
    pip install -r requirements.txt
    Download spam.csv from Kaggle and put it next to this file.

Usage:
    python spam_classifier.py                      # train + evaluate + plots
    python spam_classifier.py --predict "You won a free prize! Call now"
    python spam_classifier.py --interactive        # real-time classification
    python spam_classifier.py --no-show            # save plots without opening windows
"""

import argparse
import os
import re
import sys

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_PATH = "spam_model.joblib"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. Text preprocessing (NLTK)
# --------------------------------------------------------------------------
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()


def preprocess(text: str) -> str:
    """Lowercase, mask urls/emails/numbers, strip punctuation, remove stopwords, stem."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [STEMMER.stem(t) for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# 2. Data loading
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(
            f"Dataset not found: {path}\n"
            "Download 'spam.csv' from "
            "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset "
            "and place it next to this script (or use --data PATH)."
        )
    df = pd.read_csv(path, encoding="latin-1")
    # Kaggle version has columns v1 (label), v2 (text) + empty extras
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    else:
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["clean"] = df["text"].apply(preprocess)
    return df


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def make_pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", clf),
        ]
    )


def get_models() -> dict:
    return {
        "Naive Bayes": make_pipeline(MultinomialNB(alpha=0.1)),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000, C=10, class_weight="balanced")
        ),
        # Calibrated so SVM also gives a spam probability
        "SVM": make_pipeline(
            CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
        ),
    }


def evaluate(models, X_train, X_test, y_train, y_test):
    rows, cms, fitted = [], {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "False Positive Rate": fp / (fp + tn),
            }
        )
        cms[name] = cm
        fitted[name] = pipe
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))
    results = pd.DataFrame(rows).set_index("Model")
    return results, cms, fitted


# --------------------------------------------------------------------------
# 4. Visualizations
# --------------------------------------------------------------------------
def plot_class_distribution(df, show):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x=df["label"].map({0: "ham", 1: "spam"}), palette=["#4c9be8", "#e8654c"])
    for c in ax.containers:
        ax.bar_label(c)
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.show() if show else plt.close()


def plot_confusion_matrices(cms, show):
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
    plt.show() if show else plt.close()


def plot_model_comparison(results, show):
    ax = results[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar", figsize=(8, 5), ylim=(0.8, 1.0), rot=0
    )
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.show() if show else plt.close()


def top_spam_keywords(pipe: Pipeline, n=20):
    """Return (top spam words, top ham words) with scores from the fitted classifier."""
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    words = np.array(vec.get_feature_names_out())
    if hasattr(clf, "coef_"):  # Logistic Regression
        scores = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):  # Naive Bayes
        scores = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return None
    order = np.argsort(scores)
    spam = list(zip(words[order[::-1][:n]], scores[order[::-1][:n]]))
    ham = list(zip(words[order[:n]], scores[order[:n]]))
    return spam, ham


def plot_keywords(pipe, show, n=20):
    result = top_spam_keywords(pipe, n)
    if result is None:
        return
    spam, ham = result
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh([w for w, _ in spam][::-1], [s for _, s in spam][::-1], color="#e8654c")
    axes[0].set_title(f"Top {n} spam-indicating keywords")
    axes[1].barh([w for w, _ in ham][::-1], [abs(s) for _, s in ham][::-1], color="#4c9be8")
    axes[1].set_title(f"Top {n} ham-indicating keywords")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_keywords.png"), dpi=150)
    plt.show() if show else plt.close()


# --------------------------------------------------------------------------
# 5. Real-time classification
# --------------------------------------------------------------------------
def classify(model: Pipeline, text: str) -> dict:
    proba = model.predict_proba([text])[0][1]
    return {"label": "SPAM" if proba >= 0.5 else "HAM", "spam_probability": float(proba)}


def interactive_loop(model: Pipeline):
    print("\nReal-time spam checker. Type an email/message (or 'quit' to exit).")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        r = classify(model, text)
        print(f"  -> {r['label']}  (spam probability: {r['spam_probability']:.2%})")


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def train(data_path: str, show: bool) -> Pipeline:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(data_path)
    print(f"Loaded {len(df)} unique messages  |  spam: {df['label'].sum()}  ham: {(df['label'] == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )

    results, cms, fitted = evaluate(get_models(), X_train, X_test, y_train, y_test)
    print("\n=== Summary ===")
    print(results.round(4).to_string())
    results.round(4).to_csv(os.path.join(OUTPUT_DIR, "results.csv"))

    # Best model: lowest false-positive rate first, then highest F1
    best_name = results.sort_values(["False Positive Rate", "F1"], ascending=[True, False]).index[0]
    print(f"\nBest model (low false positives, high F1): {best_name}")

    plot_class_distribution(df, show)
    plot_confusion_matrices(cms, show)
    plot_model_comparison(results, show)
    plot_keywords(fitted["Logistic Regression"], show)

    spam_words, _ = top_spam_keywords(fitted["Logistic Regression"], 15)
    print("\nTop spam keywords:", ", ".join(w for w, _ in spam_words))

    # Wrap the best model so it accepts RAW text (preprocessing included)
    from sklearn.preprocessing import FunctionTransformer

    final = Pipeline(
        [
            ("prep", FunctionTransformer(lambda s: [preprocess(t) for t in s], validate=False)),
            ("model", fitted[best_name]),
        ]
    )
    joblib.dump({"name": best_name, "model": fitted[best_name]}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return fitted[best_name]


def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        sys.exit("No trained model found. Run `python spam_classifier.py` first.")
    return joblib.load(MODEL_PATH)["model"]


def predict_raw(model: Pipeline, text: str) -> dict:
    return classify(model, preprocess(text))


def main():
    p = argparse.ArgumentParser(description="Email Spam Classifier")
    p.add_argument("--data", default="spam.csv", help="path to spam.csv")
    p.add_argument("--predict", help="classify a single message")
    p.add_argument("--interactive", action="store_true", help="classify messages typed in the terminal")
    p.add_argument("--no-show", action="store_true", help="save plots but do not open windows")
    args = p.parse_args()

    if args.predict or args.interactive:
        model = load_model()
        if args.predict:
            r = predict_raw(model, args.predict)
            print(f"{r['label']}  (spam probability: {r['spam_probability']:.2%})")
        if args.interactive:
            print("\nReal-time spam checker. Type a message (or 'quit' to exit).")
            while True:
                try:
                    text = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() in {"quit", "exit", "q"}:
                    break
                if text:
                    r = predict_raw(model, text)
                    print(f"  -> {r['label']}  (spam probability: {r['spam_probability']:.2%})")
        return

    model = train(args.data, show=not args.no_show)

    # Quick demo of real-time classification
    print("\n=== Demo predictions ===")
    demos = [
        "WINNER!! You have won a $1000 prize. Call 09061701461 now to claim!",
        "Hey, are we still meeting for lunch tomorrow at 1?",
        "Free entry in a weekly competition. Text WIN to 80086 to enter",
        "Can you send me the notes from today's class?",
    ]
    for d in demos:
        r = predict_raw(model, d)
        print(f"[{r['label']:4}] {r['spam_probability']:.2%}  {d}")


if __name__ == "__main__":
    main()