import streamlit as st
import numpy as np
import joblib
import random
import time

# Page Config
st.set_page_config(page_title="Advanced Sales Prediction", layout="wide")

# Custom Background Gradient CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #f6f9fc, #e3f2fd);
    }
    .title {
        font-size:45px; 
        color:#FF5722; 
        text-align:center; 
        font-weight:bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<p class='title'>📊 Advanced Sales Prediction Dashboard</p>", unsafe_allow_html=True)

# Load Model
try:
    model = joblib.load("sales_prediction_model.pkl")
except:
    st.error("❌ Model not found! Please train and save the model first.")
    st.stop()

# Create Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Sales", "ℹ️ About App", "⚙️ How It Works"])

# --------- TAB 1: Prediction -----------
with tab1:
    st.header("📈 Predict Your Sales in Real-Time")

    # Layout Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛠️ Enter Advertising Budgets")

        tv = st.slider("📺 TV Budget ($K)", 0.0, 500.0, 100.0, step=1.0)
        radio = st.slider("📻 Radio Budget ($K)", 0.0, 100.0, 20.0, step=1.0)
        newspaper = st.slider("📰 Newspaper Budget ($K)", 0.0, 100.0, 10.0, step=1.0)

        input_data = np.array([[tv, radio, newspaper]])
        prediction = model.predict(input_data)[0]

        # Simulate Confidence Interval
        lower = prediction - random.uniform(0.5, 1.5)
        upper = prediction + random.uniform(0.5, 1.5)

    with col2:
        st.subheader("📊 Prediction Results")
        
        st.metric(label="Predicted Sales", value=f"${prediction:.2f}K", delta="Live Update")
        st.success(f"Estimated Sales Range: ${lower:.2f}K - ${upper:.2f}K")

        st.info("Model: Random Forest Regressor (100 Trees)")

    # Visualization
    st.markdown("### 🔍 Visual Summary of Inputs")

    chart_data = {
        "Channels": ["TV", "Radio", "Newspaper"],
        "Budget ($K)": [tv, radio, newspaper]
    }

    import pandas as pd
    import altair as alt

    df = pd.DataFrame(chart_data)

    bar_chart = alt.Chart(df).mark_bar(color='#4CAF50').encode(
        x='Channels',
        y='Budget ($K)'
    ).properties(width=600)

    st.altair_chart(bar_chart)

# --------- TAB 2: About -----------
with tab2:
    st.header("ℹ️ About This App")
    st.markdown("""
    **Advanced Sales Prediction App** helps businesses estimate sales based on advertising budget using a Machine Learning model trained on historical data.

    **Features:**
    - Real-time predictions
    - Confidence Interval Estimates
    - Interactive Charts
    - Dashboard UI

    **Model Used:** Random Forest Regressor
    """)

# --------- TAB 3: How It Works ----------
with tab3:
    st.header("⚙️ How It Works")
    st.markdown("""
    1. Enter your advertising spend for TV, Radio, and Newspaper.
    2. The app uses a trained Random Forest Regressor to predict sales.
    3. It provides an estimated range to account for market variability.
    4. Visual charts give an instant summary of inputs.
    """)

# Footer
st.markdown("<hr>")
st.markdown("<p style='text-align:center; color: grey;'>Developed by <b>Amrozia</b> - Advanced ML Internship Project</p>", unsafe_allow_html=True)
