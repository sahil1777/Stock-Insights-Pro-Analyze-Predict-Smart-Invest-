import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found for the given ticker and date range.")
        data["Name"] = yf.Ticker(ticker).info.get("longName", ticker)
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data: {e}")
