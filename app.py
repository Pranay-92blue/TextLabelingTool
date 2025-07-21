import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from textblob import TextBlob

# ----------- LOGIN SECTION -----------
def check_login():
    with st.sidebar:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
            else:
                st.error("Invalid credentials")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

check_login()
if not st.session_state.logged_in:
    st.stop()

# ----------- FILE UPLOAD -----------
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file", type="csv")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("samples.csv")

df = load_data(uploaded_file)
texts = df['text'].tolist()

# ----------- SESSION INIT -----------
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# ----------- SUGGESTION FUNCTION -----------
def get_sentiment_suggestion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

# ----------- UI SECTION -----------
st.title("Text Sentiment Labeling Tool")
st.title("📋 Text Annotation Tool - Advanced")

with st.expander("ℹ️ Annotation Instructions"):
    st.markdown("""
    **Label each sentence carefully**:
    - **Sentiment**: Is it Positive, Neutral, or Negative?
    - **Sarcasm**: Is the sentence sarcastic (ironic)?
    - **Notes**: Leave any explanation if the label is unclear.
    - Your labeling time is tracked.
    """)

# ----------- PROGRESS TRACKER -----------
total = len(texts)
labeled = len(st.session_state.annotations)
progress_pct = int((labeled / total) * 100)
st.progress(progress_pct)
st.info(f"🟢 {labeled} of {total} texts labeled.")

# ----------- LABELING LOOP -----------
index = labeled
if index < total:
    st.subheader(f"Sample {index + 1} of {total}")
    st.markdown(f"**Text:** {texts[index]}")
    
    # Suggested sentiment from ML
    suggested = get_sentiment_suggestion(texts[index])
    st.info(f"💡 Suggested Sentiment: **{suggested}** (via TextBlob)")

    sentiment = st.radio("1️⃣ Select Sentiment:", ["Positive", "Neutral", "Negative"])
    sarcasm = st.radio("2️⃣ Is this sarcastic?", ["No", "Yes"])
    comment = st.text_area("3️⃣ Any comments (optional):")

    if st.button("Submit Label"):
        elapsed = round(time.time() - st.session_state.start_time, 2)
        st.session_state.annotations.append({
            "text": texts[index],
            "sentiment": sentiment,
            "sarcasm": sarcasm,
            "comment": comment,
            "time_taken_secs": elapsed
        })
        st.session_state.start_time = time.time()

        # Autosave after each label
        backup_df = pd.DataFrame(st.session_state.annotations)
        backup_df.to_csv("annotations_autosave.csv", index=False)

        st.experimental_rerun()
else:
    st.success("✅ All texts labeled!")
    df = pd.DataFrame(st.session_state.annotations)
    st.dataframe(df)

    # Pie chart
    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    label_counts = df['sentiment'].value_counts()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Download results
    st.download_button("📥 Download Results", df.to_csv(index=False), "annotations.csv", "text/csv")


