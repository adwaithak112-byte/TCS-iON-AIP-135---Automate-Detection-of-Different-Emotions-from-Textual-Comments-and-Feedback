import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis")
    emotion = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
    return sentiment, emotion

sentiment_analyzer, emotion_detector = load_models()

# ---------------- FUNCTIONS ----------------
def detect_feedback(text):
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]

def detect_emotion(text):
    emotions = emotion_detector(text)[0]
    return sorted(emotions, key=lambda x: x["score"], reverse=True)

def plot_emotions(emotions):
    labels = [e["label"] for e in emotions]
    scores = [e["score"] for e in emotions]

    fig, ax = plt.subplots()
    ax.bar(labels, scores)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Emotion Analysis")
    st.pyplot(fig)

# ---------------- UI ----------------
st.title("Emotion & Feedback Detection System")
st.write("BCA Final Year Project – Web Application")

option = st.radio(
    "Choose Analysis Mode",
    ["Single Feedback", "Dataset Reviews (1–5)"]
)

# -------- OPTION 1 --------
if option == "Single Feedback":
    text = st.text_area("Enter user feedback")

    if st.button("Analyze Feedback"):
        if text.strip() == "":
            st.warning("Please enter some text")
        else:
            feedback, score = detect_feedback(text)
            emotions = detect_emotion(text)

            st.subheader("Sentiment Result")
            st.write(f"**{feedback}** ({score:.2f})")

            st.subheader("Emotion Scores")
            for e in emotions:
                st.write(f"{e['label']}: {e['score']:.4f}")

            plot_emotions(emotions)

# -------- OPTION 2 --------
if option == "Dataset Reviews (1–5)":
    df = pd.read_csv("review.csv")

    num = st.slider("Select number of reviews", 1, 5)

    if st.button("Analyze Dataset"):
        for i in range(num):
            review = df.iloc[i]["review"]

            st.markdown(f"### Review {i+1}")
            st.write(review)

            feedback, score = detect_feedback(review)
            emotions = detect_emotion(review)

            st.write(f"**Sentiment:** {feedback} ({score:.2f})")

            for e in emotions:
                st.write(f"{e['label']}: {e['score']:.4f}")

            plot_emotions(emotions)
