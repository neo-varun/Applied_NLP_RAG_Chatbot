import joblib
import pandas as pd
import streamlit as st
from pretrain_ai import classify_review
from preprocessing import clean_text
from gemini_api import generate_summary

st.set_page_config(page_title="Amazon Reviews EDA", layout="wide")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dataset Dashboard",
        "Rating Predictor",
        "AI Review Category",
        "AI Review Summary",
    ],
)


@st.cache_data
def load_data():
    return pd.read_csv("data/amazon_review.csv")


@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


if page == "Dataset Dashboard":

    st.title("Amazon Tech Product Reviews Dashboard")

    df = load_data()

    st.subheader("Dataset Preview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.subheader("Rating Distribution")
    st.bar_chart(df["overall"].value_counts().sort_index())

    df["review_length"] = df["reviewText"].fillna("").str.len()

    st.subheader("Review Length Distribution")
    st.bar_chart(df["review_length"].value_counts())

    st.subheader("Ratings vs Review Length")

    avg_length = df.groupby("overall")["review_length"].mean()

    st.bar_chart(avg_length)

elif page == "Rating Predictor":

    st.title("Review Rating Predictor")

    model, vectorizer = load_model()

    user_input = st.text_area("Enter a product review")

    if st.button("Predict Rating"):
        if user_input.strip() != "":
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            st.success(f"Predicted Rating: {prediction}")

            st.write("Confidence Scores:")
            for rating, prob in zip(model.classes_, probabilities):
                st.write(f"{rating}: {prob:.2f}")

        else:
            st.warning("Please enter a review.")

elif page == "AI Review Category":

    st.title("AI Review Category (Zero-Shot Model)")

    user_text = st.text_area("Enter a product review")

    if st.button("Analyze Review"):
        if user_text.strip() != "":
            label, score = classify_review(user_text)

            st.success(f"Category: {label}")
            st.info(f"Confidence: {round(score*100,2)}%")

        else:
            st.warning("Please enter a review")

elif page == "AI Review Summary":

    st.title("AI Review Summary by Gemini")

    df = load_data()

    products = df["asin"].unique()
    selected_product = st.selectbox("Choose a product", products)

    product_reviews = df[df["asin"] == selected_product]["reviewText"].dropna().head(20)
    reviews_text = "\n".join(product_reviews.tolist())

    if st.button("Generate AI Summary"):
        with st.spinner("Analyzing reviews..."):
            summary = generate_summary(reviews_text)
            st.subheader("AI pros and cons summary")
            st.write(summary)
