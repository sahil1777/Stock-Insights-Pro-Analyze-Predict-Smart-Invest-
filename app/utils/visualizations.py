import plotly.express as px
import streamlit as st

def plot_stock_data(data):

    close_column = [col for col in data.columns if "Close" in col][0]

    fig = px.line(data, x=data.index, y=close_column, title=f"{data['Name'].iloc[0]} Stock Price")
    st.plotly_chart(fig)