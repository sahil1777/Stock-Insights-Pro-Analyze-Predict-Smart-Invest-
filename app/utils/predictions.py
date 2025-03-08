import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA



def lstm_prediction(data, prediction_days):
    # Load custom CSS
    try:
        with open("app/assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure 'app/assets/styles.css' exists.")

    # Create a flex container for the header and spinner
    # st.markdown(
    #     """
    #     <div style="display: flex; align-items: center;">
    #         <h3 style="margin: 0;">LSTM Model Prediction</h3>
    #         <div id="spinner-placeholder"></div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # Add the spinner to the placeholder
    # loading_placeholder = st.empty()
    # loading_placeholder.markdown(
    #     """
    #     <div class="lds-dual-ring"></div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # Find the column name that contains 'Close'
    close_column = [col for col in data.columns if "Close" in col][0]

    # Extract the 'Close' column
    close_data = data[[close_column]].values

    # Scale the data to a range of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    # Debug: Check shapes
    # st.write("Close Data Shape:", close_data.shape)
    # st.write("Scaled Data Shape:", scaled_data.shape)

    # Ensure sufficient data for training
    sequence_length = 60  # Use 60 days of data to predict the next day
    if len(scaled_data) < sequence_length + 1:
        st.error(f"Insufficient data for LSTM training. Required: {sequence_length + 1}, Available: {len(scaled_data)}")
        return

    # Create training data
    x_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i - sequence_length:i, 0])  # Use only the first column
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Debug: Check shapes
    # st.write("x_train Shape:", x_train.shape)
    # st.write("y_train Shape:", y_train.shape)

    # Ensure x_train is a 2D array before reshaping
    if len(x_train.shape) == 1:
        x_train = np.expand_dims(x_train, axis=1)

    # Reshape x_train to be compatible with LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model (disable progress bar)
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    # Prepare test data for prediction
    test_data = scaled_data[-(sequence_length + prediction_days):]  # Use the last 60 days plus the next prediction_days
    x_test = []
    for i in range(sequence_length, len(test_data)):
        x_test.append(test_data[i - sequence_length:i, 0])  # Create overlapping sequences
    x_test = np.array(x_test)

    # Ensure x_test is a 2D array before reshaping
    if len(x_test.shape) == 1:
        x_test = np.expand_dims(x_test, axis=1)

    # Reshape x_test to be compatible with LSTM input
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Debug: Check shapes
    # st.write("Shape of x_test:", x_test.shape)
    # st.write("Number of sequences in x_test:", x_test.shape[0])

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Rescale to original values

    # Debug: Check predictions
    # st.write("Shape of predictions:", predictions.shape)
    # st.write("Length of predictions.flatten():", len(predictions.flatten()))

    # Remove the loading animation
    # loading_placeholder.empty()

    # Create a DataFrame for the predictions
    forecast_dates = pd.date_range(start=data.index[-1], periods=prediction_days + 1, freq="B")[1:]
    if len(predictions.flatten()) < prediction_days:
        st.warning(f"Only {len(predictions.flatten())} predictions available. Adjusting forecast dates.")
        forecast_dates = forecast_dates[:len(predictions.flatten())]  # Adjust forecast dates
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": predictions.flatten()})

    # Plot the predictions
    fig = px.line(forecast_df, x="Date", y="Forecast", title="LSTM Forecast")
    st.plotly_chart(fig)

    # Display the forecasted values
    st.write("**Forecasted Values:**")
    st.dataframe(forecast_df)

def arima_prediction(data, prediction_days):
    # Load custom CSS
    try:
        with open("app/assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure 'app/assets/styles.css' exists.")

    # Create a flex container for the header and spinner
    # st.markdown(
    #     """
    #     <div style="display: flex; align-items: center;">
    #         <h3 style="margin: 0;">ARIMA Model Prediction</h3>
    #         <div id="spinner-placeholder"></div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # Add the spinner to the placeholder
    # loading_placeholder = st.empty()
    # loading_placeholder.markdown(
    #     """
    #     <div class="lds-dual-ring"></div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # Find the column name that contains 'Close'
    close_column = [col for col in data.columns if "Close" in col][0]

    # Extract the 'Close' column
    close_data = data[close_column]

    # Debug: Check data
    # st.write("Close Data Shape:", close_data.shape)

    # Ensure sufficient data for training
    if len(close_data) < 1:
        st.error(f"Insufficient data for ARIMA training. Required: at least 1 data point, Available: {len(close_data)}")
        return

    # Fit the ARIMA model
    # ARIMA(p, d, q) parameters:
    # p: The number of lag observations included in the model (lag order).
    # d: The number of times the raw observations are differenced (degree of differencing).
    # q: The size of the moving average window (order of moving average).
    # You can adjust these parameters based on your data and requirements.
    model = ARIMA(close_data, order=(5, 1, 0))  # Example: ARIMA(5, 1, 0)
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.forecast(steps=prediction_days)

    # Create a DataFrame for the predictions
    forecast_dates = pd.date_range(start=data.index[-1], periods=prediction_days + 1, freq="B")[1:]
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})

    # Remove the loading animation
    # loading_placeholder.empty()

    # Plot the predictions
    fig = px.line(forecast_df, x="Date", y="Forecast", title="ARIMA Forecast")
    st.plotly_chart(fig)

    # Display the forecasted values
    st.write("**Forecasted Values:**")
    st.dataframe(forecast_df)