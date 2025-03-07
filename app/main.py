import streamlit as st
from utils.data_loader import fetch_stock_data
from utils.visualizations import plot_stock_data
from utils.predictions import lstm_prediction, arima_prediction

# Custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.title("Stock Analysis & Prediction")

# Sidebar inputs
ticker = st.text_input("Stock Symbol", placeholder="e.g., AAPL, GOOGL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if ticker and start_date and end_date:
    data = fetch_stock_data(ticker, start_date, end_date)
    if data is not None:
        st.write(f"Stock Name: {data['Name']}")
        plot_stock_data(data)

        # Tabs for additional analysis
        pricing_data, news, predict_stats = st.tabs(
            ["Pricing Data", "Top 10 News", "Prediction Stats"]
        )

        with pricing_data:
            st.write(data)

        with predict_stats:
            prediction_days = st.number_input("Days to Predict", min_value=1, max_value=100, value=30)
            if st.button("Predict"):
                lstm_prediction(data, prediction_days)
                arima_prediction(data, prediction_days)
