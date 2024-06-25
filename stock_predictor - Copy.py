# Import necessary libraries
import streamlit as st
from datetime import date

import yfinance as yf  # Library for fetching stock data
from prophet import Prophet  # Prophet forecasting model
from prophet.plot import plot_plotly  # Plotting function for Prophet
from plotly import graph_objs as go  # Plotly for interactive plots
import pandas as pd  # Data manipulation library

# Define the start date for historical data and today's date
START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set the title of the Streamlit app and introductory section
st.title('Stock Forecast App')

# Add an introductory section explaining the purpose of the app
st.markdown("""
<span style="color:#1F77B4; font-size:1.2em;">Welcome to the Stock Forecast App!</span>  
This application leverages historical stock data to forecast future stock prices using the Prophet model. Whether you're an investor looking to make informed decisions or simply curious about stock trends, this tool provides valuable insights into projected stock prices and their trends over time.
""", unsafe_allow_html=True)

# Sidebar - About the author and connect with me section
st.sidebar.title('About the author')
st.sidebar.markdown(""" 
Priyank Gupta is an Engineering student at IIT Delhi with a deep passion for Coding, AI/ML, and Analytics. His enthusiasm for technology and knowledge-sharing inspired him to create this platform, aiming to make investing both interactive and enjoyable. Whether you're here to connect, contribute, or just to say hi, Priyank is excited to have you on board!
""", unsafe_allow_html=True)

st.sidebar.subheader('Connect with Me!')
st.sidebar.text("Email: priyankg112@gmail.com")
st.sidebar.text("Instagram: @priyankgupta40")
st.sidebar.markdown("LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/itspriyankgupta)", unsafe_allow_html=True)

# List of stocks to select from
stocks = ('GOOG', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX')
# Dropdown menu for selecting the stock
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Explanation for the slider
st.markdown("""
<span style="color:#1F77B4; font-size:1.1em;">Select the number of years into the future for which you want to forecast the stock prices.</span>
""", unsafe_allow_html=True)
# Slider for selecting the number of years to predict
n_years = st.slider('Years of prediction:', 1, 3)
period = n_years * 365  # Convert years to days for Prophet forecast period

# Function to load data
@st.cache_data
def load_data(ticker):
    # Download historical stock data from Yahoo Finance
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    # Format the date to be more readable
    data['Date'] = data['Date'].dt.date
    # Drop unnecessary columns
    data = data[['Date', 'Open', 'Close', 'Volume']]
    return data

# Load data for the selected stock
data = load_data(selected_stock)

# Display the raw data
st.subheader('Raw data')
st.markdown("""
<span style="color:#1F77B4; font-size:1.1em;">Here is a preview of the raw stock data for the selected company.</span>
""", unsafe_allow_html=True)
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    # Plot 'Open' prices
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    # Plot 'Close' prices
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    # Add a title and make the range slider visible
    fig.update_layout(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Display the raw data plot
st.subheader('Stock Price Data')
st.markdown("""
<span style="color:#1F77B4; font-size:1.1em;">The following chart shows the historical stock prices (open and close) with an interactive range slider.</span>
""", unsafe_allow_html=True)
plot_raw_data()

# Prepare data for Prophet model
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Train the Prophet model
m = Prophet()
m.fit(df_train)
# Create a future dataframe for predictions
future = m.make_future_dataframe(periods=period)

# Ensure 'Date' column in future dataframe is in datetime format
future['ds'] = pd.to_datetime(future['ds'].dt.date)

# Predict using the Prophet model
forecast = m.predict(future)

# Convert last_date to pandas Timestamp to ensure consistency
last_date = pd.Timestamp(data['Date'].iloc[-1])

# Filter forecast data to start from the day after the last historical date
forecast_filtered = forecast[forecast['ds'] > last_date]

# Format forecast data for readability
forecast_display = forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_display = forecast_display.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"})

# Remove time portion from 'Date' column
forecast_display['Date'] = forecast_display['Date'].dt.date

# Display forecast data
st.subheader('Forecast data')
st.markdown("""
<span style="color:#1F77B4; font-size:1.1em;">Here is a preview of the forecasted stock data.</span>
""", unsafe_allow_html=True)
st.write(forecast_display)

# Plot the forecast
st.subheader(f'Forecast plot for {n_years} years')
st.markdown(f"""
<span style="color:#1F77B4; font-size:1.1em;">The following chart shows the predicted stock prices for the next {n_years} years.</span>
""", unsafe_allow_html=True)
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Display forecast components
st.subheader("Forecast components")
st.markdown("""
<span style="color:#1F77B4; font-size:1.1em;">The following charts show the individual components (trend, weekly, yearly) of the forecast.</span>
""", unsafe_allow_html=True)
fig2 = m.plot_components(forecast)
st.write(fig2)
