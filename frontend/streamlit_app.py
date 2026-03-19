import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title = "Twitter Sentiment Analysis",
    page_icon  = "T",
    layout     = "centered"
)

st.title("Twitter Sentiment Analysis")
st.markdown("Enter a tweet to analyze its sentiment.")

tweet  = st.text_area(label = "Enter tweet", height=100)
button = st.button(label = "Predict")

if button:
    if not tweet.strip():
        st.warning("Please enter a tweet first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict/",
                    json = {"tweet": tweet}
                )

                if response.status_code == 200:
                    result = response.json()

                    sentiment = result["Prediction"]
                    confidence = result["Confidence_Score"]

                    if sentiment == "Positive":
                        st.success(f"Positive")
                    elif sentiment == "Neutral":
                        st.warning(f"Neutral")
                    else:
                        st.error(f"Negative")

                    st.write(f"Confidence Score: {confidence}%")

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Connection Error: {e}")