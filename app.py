import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import (
    add_prediction_details,
    view_all_prediction_details,
    create_emotionclf_table,
    IST
)

@st.cache_resource
def load_model():
    return joblib.load("./models/emotion_classifier_pipe_lr.pkl")

@st.cache_resource
def init_db():
    create_emotionclf_table()

pipe_lr = load_model()

def predict_emotions(text):
    return pipe_lr.predict([text])[0]

def get_prediction_proba(text):
    return pipe_lr.predict_proba([text])

emotions_emoji_dict = {
    "anger": "Angry",
    "disgust": "Disgusting",
    "fear": "Fear",
    "happy": "Happy",
    "joy": "Joy",
    "neutral": "Neutral",
    "sad": "Sad",
    "sadness": "Sadness",
    "shame": "Shame",
    "surprise": "Surprise"
}

def emotion_app():
    hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
.st-emotion-cache-scp8yw {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("🧠 Emotion Detection System")

    menu = ["Home", "History"]
    choice = st.sidebar.selectbox("Menu", menu)

    init_db()

    if choice == "Home":
        st.subheader("Emotion Detection from Text")

        with st.form(key="emotion_form"):
            raw_text = st.text_area("Enter text")
            submit = st.form_submit_button("Analyze")

        if submit:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(
                raw_text,
                prediction,
                np.max(probability),
                datetime.now(IST)
            )

            with col1:
                st.header("Result")
                st.write(raw_text)
                st.success(f"{prediction} {emotions_emoji_dict[prediction]}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                proba_df = pd.DataFrame(
                    probability,
                    columns=pipe_lr.classes_
                ).T.reset_index()
                proba_df.columns = ["Emotion", "Probability"]

                fig = alt.Chart(proba_df).mark_bar().encode(
                    x="Emotion",
                    y="Probability",
                    color="Emotion"
                )
                st.altair_chart(fig, use_container_width=True)

    elif choice == "History":
        st.subheader("Prediction History")

        df = pd.DataFrame(
            view_all_prediction_details(),
            columns=["Text", "Emotion", "Confidence", "Time"]
        )

        st.dataframe(df)
        
if __name__ == "__main__":
    emotion_app()
