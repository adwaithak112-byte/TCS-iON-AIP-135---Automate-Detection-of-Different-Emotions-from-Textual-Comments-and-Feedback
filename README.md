# Automate Detection of Different Emotions from Textual Comments

A Streamlit-based Emotion & Feedback Detection System that predicts:
- Sentiment (Positive / Negative / Neutral / Mixed)
- Emotion (Joy, Sadness, Anger, Fear, Disgust, Surprise, Neutral)

## Features
- Single review analysis
- Dataset (CSV) upload + filtering
- Emotion model: fallback + optional custom model
- Visual confidence bars + charts

## Tech Stack
- Python
- Streamlit
- Transformers (Hugging Face)
- Pandas, Matplotlib

## Dataset
Use any CSV with a required column named: **review**
Optional: **id**

Example format:
| id | review |
|----|--------|
| 1  | it was a good movie and i liked it |

## How to Run
1) Create venv + install requirements:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
