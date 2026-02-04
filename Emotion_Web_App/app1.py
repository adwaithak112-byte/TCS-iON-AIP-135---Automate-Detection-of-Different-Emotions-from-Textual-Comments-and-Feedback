import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="Emotion Detection App", page_icon="🧠", layout="centered")

emotion_emoji = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😱",
    "disgust": "🤢",
    "surprise": "😲",
    "neutral": "😐",
}

sentiment_emoji = {
    "POSITIVE": "👍😊",
    "NEGATIVE": "👎😞",
    "NEUTRAL": "😐",
}

def safe_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def normalize_emotion_output(result):

    if isinstance(result, dict):
        return [result]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        return result
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        return result[0]
    return [{"label": "neutral", "score": 1.0}]

st.markdown("## 🧠 Emotion & Feedback Detection System")
st.markdown("---")

@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    # IMPORTANT: top_k=None returns ALL emotion labels
    emotion = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    return sentiment, emotion

with st.spinner("⏳ Loading AI models... Please wait"):
    sentiment_analyzer, emotion_detector = load_models()
st.success("✅ Models loaded successfully!")

def detect_sentiment(text: str):
    text = safe_text(text)
    if not text:
        return "NEUTRAL", 0.0
    r = sentiment_analyzer(text)[0]
    return r.get("label", "NEUTRAL"), float(r.get("score", 0.0))

def detect_emotions(text: str):
    text = safe_text(text)
    if not text:
        return [{"label": "neutral", "score": 1.0}]

    raw = emotion_detector(text)
    emotions = normalize_emotion_output(raw)

    emotions = [e for e in emotions if isinstance(e, dict) and "label" in e and "score" in e]
    if not emotions:
        emotions = [{"label": "neutral", "score": 1.0}]

    emotions = sorted(emotions, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return emotions

def show_all_emotions(emotions):
    """Table + % + emojis + progress bars for ALL emotions."""
    rows = []
    for e in emotions:
        label = safe_text(e["label"]).lower()
        score = float(e["score"])
        rows.append({
            "Emoji": emotion_emoji.get(label, "🙂"),
            "Emotion": label.upper(),
            "Score (0–1)": round(score, 4),
            "Percent (%)": f"{score * 100:.1f}%"
        })

    df_em = pd.DataFrame(rows).sort_values("Score (0–1)", ascending=False)

    st.subheader("📌 All Emotions (Detailed)")
    st.dataframe(df_em, use_container_width=True)

    st.subheader("📈 Emotion Confidence Bars")
    for _, r in df_em.iterrows():
        st.write(f"{r['Emoji']} **{r['Emotion']}** — {r['Percent (%)']}")
        st.progress(min(int(float(r["Score (0–1)"]) * 100), 100))

def plot_emotions(emotions, title="Emotion Distribution 📊"):
    """Detailed chart: sorted + value labels (score + %) + grid."""
    emotions_sorted = sorted(emotions, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    labels = [safe_text(e["label"]).lower() for e in emotions_sorted]
    scores = [float(e["score"]) for e in emotions_sorted]
    perc = [s * 100 for s in scores]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, scores)

    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Confidence (0–1)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, s, p in zip(bars, scores, perc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{s:.3f} ({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

menu = st.radio("Choose Analysis Type", ["✍️ Single Review", "📂 Dataset Analysis"])

if menu == "✍️ Single Review":
    review_text = st.text_area("📝 Enter your review/feedback:")

    if st.button("🔍 Analyze Review"):
        review_text = safe_text(review_text)
        if not review_text:
            st.warning("⚠️ Please enter some text.")
        else:
            sentiment, sent_score = detect_sentiment(review_text)
            emotions = detect_emotions(review_text)

            top_emotion = safe_text(emotions[0]["label"]).lower()
            top_score = float(emotions[0]["score"])

            st.markdown(f"## {sentiment_emoji.get(sentiment,'🙂')} Sentiment: **{sentiment}**")
            st.write(f"Confidence: **{sent_score:.2f}**")

            st.markdown(f"## {emotion_emoji.get(top_emotion,'🙂')} Top Emotion: **{top_emotion.upper()}**")
            st.write(f"Top Emotion Confidence: **{top_score:.2f}**")

            show_all_emotions(emotions)
            plot_emotions(emotions, "Emotion Distribution (All Emotions) 📊")

else:
    st.info("📌 Upload your CSV (required column: **review**). Optional column: **id**.")
    uploaded_file = st.file_uploader("📂 Drag & drop your CSV here", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Validate required column
        if "review" not in df.columns:
            st.error("❌ Your CSV must contain a column named **review**.")
            st.stop()

        # Create id if missing
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)

        # Clean reviews
        df["review"] = df["review"].apply(safe_text)
        df = df[df["review"] != ""].copy()

        if df.empty:
            st.error("❌ No valid reviews found (empty/NaN). Please upload a proper dataset.")
            st.stop()

        st.success("✅ Dataset uploaded successfully!")
        st.write("📄 Dataset Preview (first 20 rows):")
        st.dataframe(df[["id", "review"]].head(20), use_container_width=True)

        # Compute sentiment for all rows (so filtering works)
        with st.spinner("🔎 Calculating sentiment for dataset..."):
            sentiments = []
            scores = []
            for text in df["review"]:
                lab, sc = detect_sentiment(text)
                sentiments.append(lab)
                scores.append(sc)

        df["Sentiment"] = sentiments
        df["SentimentScore"] = scores
        df["SentimentEmoji"] = df["Sentiment"].map(lambda x: sentiment_emoji.get(x, "🙂"))

        # Filter selection
        filter_choice = st.selectbox("Filter reviews by sentiment", ["All", "Positive", "Negative"])

        if filter_choice == "Positive":
            df_filtered = df[df["Sentiment"] == "POSITIVE"].copy()
        elif filter_choice == "Negative":
            df_filtered = df[df["Sentiment"] == "NEGATIVE"].copy()
        else:
            df_filtered = df.copy()

        if df_filtered.empty:
            st.warning("⚠️ No reviews match this filter. Try selecting **All**.")
            st.stop()

        st.subheader("✅ Filtered Reviews")
        st.dataframe(
            df_filtered[["id", "SentimentEmoji", "Sentiment", "SentimentScore", "review"]],
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("🎯 Analyze Specific Review")

        id_list = df_filtered["id"].tolist()
        selected_id = st.selectbox("Select Review ID", id_list)

        if st.button("🔍 Analyze Selected Review"):
            row = df_filtered[df_filtered["id"] == selected_id].iloc[0]
            selected_review = row["review"]

            sentiment, sent_score = detect_sentiment(selected_review)
            emotions = detect_emotions(selected_review)

            top_emotion = safe_text(emotions[0]["label"]).lower()
            top_score = float(emotions[0]["score"])

            st.markdown(f"### 🧾 Review ID: {selected_id}")
            st.write(selected_review)

            st.markdown(f"## {sentiment_emoji.get(sentiment,'🙂')} Sentiment: **{sentiment}**")
            st.write(f"Confidence: **{sent_score:.2f}**")

            st.markdown(f"## {emotion_emoji.get(top_emotion,'🙂')} Top Emotion: **{top_emotion.upper()}**")
            st.write(f"Top Emotion Confidence: **{top_score:.2f}**")

            show_all_emotions(emotions)
            plot_emotions(emotions, title=f"Emotion Distribution for Review ID {selected_id} 📊")

        st.markdown("---")
        st.subheader("📊 Optional: Analyze Multiple Filtered Reviews")

        max_n = min(10, len(df_filtered))
        n = st.slider("How many reviews to analyze (from filtered list)?", 1, max_n, 5)

        if st.button("🚀 Analyze First N Filtered Reviews"):
            for i in range(n):
                r = df_filtered.iloc[i]
                rid = r["id"]
                txt = r["review"]

                sentiment, sent_score = detect_sentiment(txt)
                emotions = detect_emotions(txt)

                top_emotion = safe_text(emotions[0]["label"]).lower()
                top_score = float(emotions[0]["score"])

                st.markdown(f"### 🧾 Review ID: {rid}")
                st.write(txt)
                st.write(f"**{sentiment_emoji.get(sentiment,'🙂')} Sentiment:** {sentiment} ({sent_score:.2f})")
                st.write(f"**{emotion_emoji.get(top_emotion,'🙂')} Top Emotion:** {top_emotion.upper()} ({top_score:.2f})")

                show_all_emotions(emotions)
                plot_emotions(emotions, title=f"Emotion Distribution for Review ID {rid} 📊")

st.markdown("---")
st.markdown("👨‍🎓 **College Mini Project | Emotion Detection from Text**")

