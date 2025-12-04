import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess_text, preprocess_series


# --- Load JSON data with optional balancing ---
def load_json(path="News_Category_Dataset_v3.json", max_per_category=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    df = pd.read_json(path, lines=True)
    rename_map = {"headline": "title", "short_description": "description", "category": "category"}
    df = df.rename(columns=rename_map)
    df = df[["title", "description", "category"]].dropna()
    df["category"] = df["category"].astype(str)
    df["text"] = (df["title"] + " " + df["description"]).apply(preprocess_text)

    if max_per_category is not None:
        # Баланслах: category бүрээс max_per_category ширхэг авна
        df = df.groupby("category").apply(
            lambda x: x.sample(min(len(x), max_per_category), random_state=42)
        ).reset_index(drop=True)

    return df

# --- Build pipeline ---
def build_pipeline(model_name="lr", ngram_max=2, min_df=2, max_df=0.95):
    preproc = FunctionTransformer(preprocess_series)  # named function, not lambda
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, ngram_max),
                                 min_df=min_df, max_df=max_df)
    if model_name == "nb":
        clf = MultinomialNB()
    elif model_name == "lr":
        clf = LogisticRegression(max_iter=1000)
    elif model_name == "svm":
        clf = LinearSVC()
    else:
        raise ValueError("Choose model from: nb, lr, svm")
    return Pipeline([
        ("preproc", preproc),
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

# --- Confusion matrix ---
def plot_cm(cm, labels):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# --- Training ---
def train(json_path="News_Category_Dataset_v3.json",
          model_name="lr",
          save_path="artifacts/model.joblib",
          max_per_category=1000):
    
    df = load_json(json_path, max_per_category=max_per_category)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2, random_state=42,
        stratify=df["category"]
    )

    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(df["category"].unique()))
    plot_cm(cm, sorted(df["category"].unique()))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(pipeline, save_path)
    print(f"Model saved to {save_path}")

# --- Run ---
if __name__ == "__main__":
    train()
