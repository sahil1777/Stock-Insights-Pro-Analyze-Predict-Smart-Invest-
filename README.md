# Stock Insights Pro: Analyze, Predict & Smart Invest

Stock Insights Pro is a cutting-edge application designed to empower investors with advanced tools for real time stock market analysis, price prediction, and AI-driven investment recommendations. By leveraging state-of-the-art machine learning models, natural language processing (NLP), and comprehensive financial data integration, Stock Insights Pro provides users with actionable insights to make informed investment decisions.


## Features
- Stock Data Visualization: Interactive charts for historical stock prices.
- Financial Statements: Integration of income statements, balance sheets, and cash flow statements.
- Price Prediction:
  - LSTM Model: Deep learning-based stock price prediction.
  - ARIMA Model: Statistical time series forecasting.
- Sentiment Analysis:
  - VADER: Rule-based sentiment analysis for news articles.
  - BERT: Advanced NLP-based sentiment analysis.
- AI Expert Opinion: Investment recommendations powered by Google Gemini.
- User-Friendly Interface: Built with Streamlit for an intuitive and interactive experience.


## Installation

Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning the repository)

Steps

1. Clone the Repository:

```bash
  git clone https://github.com/sahil1777/Stock-Insights-Pro-Analyze-Predict-Smart-Invest-.git
  cd Stock_App
```

2. Set Up a Virtual Environment (recommended):
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install Dependencies:
```bash
pip install -r requirements.txt
```

4. Set Up API Keys:
- Create a .env file in the root directory and add your API keys:
  ```
  NEWS_API_KEY=your_newsapi_key
  FMP_API_KEY=your_fmp_api_key
  GEMINI_API_KEY=your_gemini_api_key
  ```

5. Run the App:
```bash
streamlit run app/main.py
  ```

6. Access the App:
- Open your browser and navigate to http://localhost:8501.
    
## Usage/Examples

```javascript
- Enter Stock Symbol: Input the stock ticker (e.g., AAPL, GOOGL).
- Select Date Range: Choose the start and end dates for analysis.
- Explore Features:
    - View historical stock data and trends.
    - Access financial statements (income statement, balance sheet, cash flow statement).
    - Analyze sentiment from news articles.
    - Predict future stock prices using LSTM or ARIMA models. 
    - Get AI-driven investment recommendations.  
```


## Project Structure

```javascript
Stock_App/
├── app/
│   ├── main.py                  # Main Streamlit application
│   ├── assets/
│   │   └── styles.css           # Custom CSS for the app
├── utils/
│   ├── data_loader.py           # Fetch stock data and news
│   ├── predictions.py           # LSTM and ARIMA models
│   ├── sentiment_analysis.py    # VADER and BERT sentiment analysis
│   ├── financial_data.py        # Fetch financial statements
│   ├── gemini_integration.py    # Google Gemini integration
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
├── README.md                    # Project documentation
└── screenshots/                 # Screenshots for the README
```
## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
```javascript
1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.
```


## Acknowledgements

- Streamlit for the amazing web framework.
- Yahoo Finance and Financial Modeling Prep for financial data.
- NewsAPI for news articles.
- Google Gemini for AI-driven insights.
- Hugging Face for the BERT model.


## Contact

For questions or feedback, feel free to reach out:

- Name: Sahil Naik
- Email: sahilnaik1709@gmail.com
- GitHub: sahil1777
- LinkedIn: https://www.linkedin.com/in/sahil-naik-47432918b/