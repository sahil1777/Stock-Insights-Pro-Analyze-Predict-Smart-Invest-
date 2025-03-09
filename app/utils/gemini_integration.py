import google.generativeai as genai

# Initialize Gemini
def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    return model

def get_ai_expert_opinion(model, stock_data, sentiment_analysis, financial_data, risk_tolerance, investment_horizon):
    # Prepare the prompt
    prompt = f"""
    You are a financial expert analyzing a stock for investment. Based on the following data, provide a recommendation (Buy, Hold, or Sell), a confidence level (0-100%), and a detailed reasoning.

    Stock Data:
    - Ticker: {stock_data['ticker']}
    - Historical Prices: {stock_data['historical_prices']}
    - Recent Trends: {stock_data['recent_trends']}

    Sentiment Analysis:
    - Overall Sentiment Score: {sentiment_analysis['overall_score']}
    - Sentiment Distribution: {sentiment_analysis['sentiment_distribution']}

    Financial Data:
    - Revenue (Latest): {financial_data['income_statement']['revenue'].iloc[0] if financial_data['income_statement'] is not None else 'N/A'}
    - Net Income (Latest): {financial_data['income_statement']['netIncome'].iloc[0] if financial_data['income_statement'] is not None else 'N/A'}
    - Total Assets (Latest): {financial_data['balance_sheet']['totalAssets'].iloc[0] if financial_data['balance_sheet'] is not None else 'N/A'}
    - Total Liabilities (Latest): {financial_data['balance_sheet']['totalLiabilities'].iloc[0] if financial_data['balance_sheet'] is not None else 'N/A'}
    - Operating Cash Flow (Latest): {financial_data['cash_flow']['operatingCashFlow'].iloc[0] if financial_data['cash_flow'] is not None else 'N/A'}

    User Inputs:
    - Risk Tolerance: {risk_tolerance}
    - Investment Horizon: {investment_horizon}

    Provide your recommendation in the following format:
    Recommendation: [Buy/Hold/Sell]
    Confidence Level: [X%]
    Reasoning: [Detailed explanation]
    """

    # Get the AI response
    response = model.generate_content(prompt)
    return response.text