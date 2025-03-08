import yfinance as yf
import pandas as pd 
import requests
import streamlit as st

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found for the given ticker and date range.")
        
        # Flatten MultiIndex column names
        if isinstance(data.columns, pd.MultiIndex):  # Ensure 'pd' is defined
            data.columns = [' '.join(col).strip() for col in data.columns.values]
        
        # Add stock name to the DataFrame
        data["Name"] = yf.Ticker(ticker).info.get("longName", ticker)
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data: {e}")
    

def fetch_stock_news(ticker, api_key):
    """
    Fetch top 10 news articles related to the given stock ticker using NewsAPI.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,  # Search query (stock ticker)
        "apiKey": api_key,  # Your NewsAPI key
        "pageSize": 10,  # Number of articles to fetch
        "sortBy": "publishedAt",  # Sort by publication date (latest first)
        "language": "en",  # Filter for English articles
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return articles
    else:
        st.error(f"Failed to fetch news. Error: {response.status_code}")
        return []


def fetch_financial_statements(api_key, ticker):
    base_url = "https://financialmodelingprep.com/api/v3"
    endpoints = {
        "income_statement": f"{base_url}/income-statement/{ticker}?apikey={api_key}",
        "balance_sheet": f"{base_url}/balance-sheet-statement/{ticker}?apikey={api_key}",
        "cash_flow": f"{base_url}/cash-flow-statement/{ticker}?apikey={api_key}",
    }

    financial_data = {}
    for statement_type, url in endpoints.items():
        response = requests.get(url)
        if response.status_code == 200:
            # Convert JSON to DataFrame
            financial_data[statement_type] = pd.DataFrame(response.json())
        else:
            st.error(f"Failed to fetch {statement_type.replace('_', ' ')} for {ticker}.")
            financial_data[statement_type] = None

    return financial_data