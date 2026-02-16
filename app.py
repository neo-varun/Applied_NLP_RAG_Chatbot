import streamlit as st
import pandas as pd

st.set_page_config(page_title="Amazon Reviews EDA", layout="wide")

st.title("Amazon Tech Product Reviews Dashboard")

st.sidebar.header("Navigation")
st.sidebar.write("Day 1: Dataset Overview")


@st.cache_data
def load_data():
    return pd.read_csv("data/amazon_review.csv")


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
