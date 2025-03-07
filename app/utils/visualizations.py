import plotly.express as px

def plot_stock_data(data):
    fig = px.line(data, x=data.index, y="Close", title=f"{data['Name']} Stock Price")
    st.plotly_chart(fig) 
