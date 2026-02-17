import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessing import clean_text

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/amazon_review.csv")

df["reviewText"] = df["reviewText"].fillna("")

df["cleaned"] = df["reviewText"].apply(clean_text)

X = df["cleaned"]
y = df["overall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train_tfidf, y_train)

preds = model.predict(X_test_tfidf)

print(classification_report(y_test, preds))

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model successfully trained and saved")
