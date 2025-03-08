import openai

# Initialize OpenAI
def initialize_openai(api_key):
    openai.api_key = api_key

# Get AI Expert Opinion
def get_ai_expert_opinion(stock_data, sentiment_analysis, risk_tolerance, investment_horizon):
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

    User Inputs:
    - Risk Tolerance: {risk_tolerance}
    - Investment Horizon: {investment_horizon}

    Provide your recommendation in the following format:
    Recommendation: [Buy/Hold/Sell]
    Confidence Level: [X%]
    Reasoning: [Detailed explanation]
    """

    # Get the AI response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5 or GPT-4
        messages=[
            {"role": "system", "content": "You are a financial expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content