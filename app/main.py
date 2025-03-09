
import os
import streamlit as st
import datetime
from dotenv import load_dotenv
from utils.data_loader import fetch_stock_data, fetch_stock_news, fetch_financial_statements
from utils.visualizations import plot_stock_data
from utils.predictions import lstm_prediction, arima_prediction
from utils.sentiment_analysis import vader_sentiment_analysis, bert_sentiment_analysis, create_sentiment_distribution_chart
from utils.gemini_integration import initialize_gemini, get_ai_expert_opinion


try:
    with open("app/assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("CSS file not found. Please ensure 'app/assets/styles.css' exists.")

st.markdown(
    """
    <h2 style='text-align: left;'>
        <span style='color: #2E86C1;'>Stock Insights Pro:</span>
        <span style='color: #E67E22;'>Analyze-Predict-Invest</span>
    </h2>
    """,
    unsafe_allow_html=True,
)

# Sidebar inputs
ticker = st.text_input("Stock Symbol", placeholder="e.g., AAPL, GOOGL")
today = datetime.date.today()
default_start_date = today - datetime.timedelta(days=180)  # Approximate 6 months

start_date = st.date_input("Start Date", value=default_start_date)
# start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if start_date >= end_date:
    st.error("Start date must be before the end date.")


# Initialize session state
if "proceed_clicked" not in st.session_state:
    st.session_state.proceed_clicked = False
if "active_model" not in st.session_state:
    st.session_state.active_model = None

load_dotenv()

# Access the API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if ticker and start_date and end_date:
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data is not None:
        plot_stock_data(data)

        pricing_data, financial_statements, news_sentiment_analysis, predict_stats, ai_expert_opinion = st.tabs(
            ["Pricing Data", "Financial Statements", "News Sentiment Analysis", "Prediction Stats", "AI Expert Opinion"]
        )
        with pricing_data:
            with st.spinner("Getting Pricing data..."):
                st.subheader(f"Pricing data for {ticker}")
                st.write(data)

        with news_sentiment_analysis:
            st.subheader(f"News Sentiment Analysis for {data['Name'].iloc[0]}")
            if NEWS_API_KEY:
                articles = fetch_stock_news(ticker, NEWS_API_KEY)
                if articles:
                    # Perform sentiment analysis
                    vader_results, overall_score, overall_sentiment = vader_sentiment_analysis(articles)
                    bert_results = bert_sentiment_analysis(articles)

                    if overall_score > 0.05:
                        color = "green"
                    elif overall_score < -0.05:
                        color = "red"
                    else:
                        color = "blue"

                    st.markdown(
                        f"**Overall Sentiment Score:** <span style='color: {color}; font-weight: bold; font-size: 20px;'>{overall_score:.2f}</span>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"**Overall Sentiment:** <span style='color: {color}; font-weight: bold; font-size: 20px;'>{overall_sentiment}</span>",
                        unsafe_allow_html=True
                    )
                    st.progress((overall_score + 1) / 2)  # Normalize to 0-1 range

                    # Display sentiment distribution chart
                    st.plotly_chart(create_sentiment_distribution_chart(vader_results))

                    # Display individual articles with sentiment analysis
                    for i, (article, vader_result, bert_result) in enumerate(zip(articles, vader_results, bert_results), 1):
                        with st.expander(f"### {i}. {article['title']}"):
                            st.write(f"**Source:** {article['source']['name']}")
                            st.write(f"**Published At:** {article['publishedAt']}")
                            st.write(f"**Description:** {article['description']}")
                            st.write(f"[Read more]({article['url']})")
                            
                           
                            st.write("**Sentiment Analysis**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**VADER Sentiment**")
                                compound_score = vader_result['sentiment']['compound']
                                if compound_score > 0.05:
                                    color = "green"
                                elif compound_score < -0.05:
                                    color = "red"
                                else:
                                    color = "blue"
                                
                                st.markdown(
                                    f"**Compound Score:** <span style='color: {color}; font-weight: bold;'>{compound_score:.2f}</span>",
                                    unsafe_allow_html=True
                                )
        
                                st.write(f"Sentiment: {vader_result['sentiment']}")
                            with col2:
                                st.write("**BERT Sentiment**")
                                compound_score = bert_result['sentiment']['compound']
                                if compound_score > 0.05:
                                    color = "green"
                                elif compound_score < -0.05:
                                    color = "red"
                                else:
                                    color = "blue"
                                
                                st.markdown(
                                    f"**Compound Score:** <span style='color: {color}; font-weight: bold;'>{compound_score:.2f}</span>",
                                    unsafe_allow_html=True
                                )
                                
                                sentiment_label = bert_result['sentiment']['label']
                                if sentiment_label == "Positive":
                                    color = "green"
                                elif sentiment_label == "Negative":
                                    color = "red"
                                else:
                                    color = "blue"

                               
                                st.markdown(
                                    f"**Sentiment:** <span style='color: {color}; font-weight: bold;'>{sentiment_label}</span> (Confidence: {bert_result['sentiment']['confidence']:.2f})",
                                    unsafe_allow_html=True
    )

                            st.write("---")
                else:
                    st.warning("No news articles found for this stock.")
            else:
                st.error("NewsAPI key is missing. Please provide a valid API key.")



        with predict_stats:
            prediction_days = st.number_input("Days to Predict", min_value=1, max_value=50, value=10)

            if st.button("Proceed"):
                st.session_state.proceed_clicked = True

            if st.session_state.proceed_clicked:
                model_1, model_2 = st.tabs(["LSTM Model Prediction", "ARIMA Model Prediction"])

                with model_1:
                    if st.button("Predict", key="predict_lstm"):
                        st.session_state.active_model = "LSTM"
                    if st.session_state.active_model == "LSTM":
                        with st.spinner("Training LSTM model and making predictions..."):
                            lstm_prediction(data, prediction_days)

                with model_2:
                    if st.button("Predict", key="predict_arima"):
                        st.session_state.active_model = "ARIMA"
                    if st.session_state.active_model == "ARIMA":
                        with st.spinner("Fitting ARIMA model and making predictions..."):
                            arima_prediction(data, prediction_days)

        with ai_expert_opinion:
            st.subheader("AI Expert Opinion")
            if GEMINI_API_KEY:
                gemini_model = initialize_gemini(GEMINI_API_KEY)

                # User inputs for risk tolerance and investment horizon
                risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
                investment_horizon = st.selectbox("Investment Horizon", ["Short-term", "Medium-term", "Long-term"])

                close_column = [col for col in data.columns if "Close" in col][0]

                close_data = data[close_column]

                stock_data = {
                    "ticker": ticker,
                    "historical_prices": close_data.tolist(),  # Convert to list for Gemini input
                    "recent_trends": "Upward" if close_data.iloc[-1] > close_data.iloc[-30] else "Downward"
                }
                sentiment_analysis = {
                    "overall_score": overall_score,
                    "sentiment_distribution": "Positive: 60%, Negative: 20%, Neutral: 20%"  
                }

                financial_data = fetch_financial_statements(FMP_API_KEY, ticker)

                # Get AI Expert Opinion
                if st.button("Get AI Expert Opinion"):
                    with st.spinner("Analyzing stock data and generating recommendation..."):
                        ai_opinion = get_ai_expert_opinion(gemini_model, stock_data, sentiment_analysis, financial_data, risk_tolerance, investment_horizon)
                        st.write(ai_opinion)
            else:
                st.error("Gemini API key is missing. Please provide a valid API key.")


        with financial_statements:
            with st.spinner("Getting Financial..."):
                st.subheader(f"Financial Statements for {ticker}")
                if FMP_API_KEY:
                    financial_data = fetch_financial_statements(FMP_API_KEY, ticker)
                    if financial_data:
                        st.write("### Income Statement")
                        if financial_data["income_statement"] is not None:
                            st.dataframe(financial_data["income_statement"])
                        else:
                            st.warning("No income statement data available.")

                        st.write("### Balance Sheet")
                        if financial_data["balance_sheet"] is not None:
                            st.dataframe(financial_data["balance_sheet"])
                        else:
                            st.warning("No balance sheet data available.")

                        st.write("### Cash Flow Statement")
                        if financial_data["cash_flow"] is not None:
                            st.dataframe(financial_data["cash_flow"])
                        else:
                            st.warning("No cash flow statement data available.")
                    else:
                        st.error("Failed to fetch financial statements.")
                else:
                    st.error("Financial Modeling Prep API key is missing. Please provide a valid API key.") if FMP_API_KEY else ""