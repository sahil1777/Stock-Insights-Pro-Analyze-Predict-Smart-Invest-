import numpy as np
import pandas as pd
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Initialize VADER
def initialize_vader():
    sia = SentimentIntensityAnalyzer()
    return sia

# Perform VADER sentiment analysis
def vader_sentiment_analysis(articles):
    sia = initialize_vader()
    results = []
    compound_scores = []  # Store compound scores for aggregation
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        text = f"{title}. {description}"
        sentiment = sia.polarity_scores(text)
        results.append({
            "title": title,
            "description": description,
            "sentiment": sentiment
        })
        compound_scores.append(sentiment["compound"])  # Add compound score to the list
    
    # Calculate overall sentiment score
    overall_score = np.mean(compound_scores) if compound_scores else 0
    overall_sentiment = "Positive" if overall_score > 0.05 else "Negative" if overall_score < -0.05 else "Neutral"
    
    return results, overall_score, overall_sentiment


# Initialize BERT sentiment analysis pipeline
def initialize_bert():
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_pipeline

# Perform BERT sentiment analysis
def bert_sentiment_analysis(articles):
    sentiment_pipeline = initialize_bert()
    results = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        text = f"{title}. {description}"
        sentiment = sentiment_pipeline(text)[0]
        
        # Map BERT's output to a VADER-like format
        if sentiment["label"] == "POSITIVE":
            compound_score = sentiment["score"]  # Positive sentiment
        else:
            compound_score = -sentiment["score"]  # Negative sentiment
        
        # Determine sentiment label
        if compound_score > 0.05:
            sentiment_label = "Positive"
        elif compound_score < -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        results.append({
            "title": title,
            "description": description,
            "sentiment": {
                "compound": compound_score,
                "label": sentiment_label,
                "confidence": sentiment["score"]
            }
        })
    return results




# Create sentiment distribution chart
def create_sentiment_distribution_chart(vader_results):
    sentiment_labels = []
    for result in vader_results:
        compound_score = result["sentiment"]["compound"]
        if compound_score > 0.05:
            sentiment_labels.append("Positive")
        elif compound_score < -0.05:
            sentiment_labels.append("Negative")
        else:
            sentiment_labels.append("Neutral")
    
    # Count sentiment distribution
    sentiment_counts = pd.Series(sentiment_labels).value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    # Define colors for each sentiment
    color_map = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "gray"
    }
    
    # Create a pie chart
    fig = px.pie(
        sentiment_counts,
        values="Count",
        names="Sentiment",
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        autosize=False,
        width=150,  
        height=150,  
        margin=dict(l=20, r=20, t=30, b=20)  
    )
    
    return fig