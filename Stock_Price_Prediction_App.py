import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bsedata.bse import BSE
from keras.models import load_model
import pytz

# App Title

st.title("Stock Price Predict App")

st.sidebar.subheader("Query Parameters")
start = st.sidebar.date_input("Start date", dt.datetime(2010, 1, 1))
end = st.sidebar.date_input("End date", dt.datetime.today())
# end = st.sidebar.date_input("End date", dt.datetime.today())

# Get user input
user_input = st.text_input("Enter Stock Ticker",'HDFC.NS')

# Retrieve stock data using yfinance
df = yf.download(user_input, start=start, end=end)
# Reset index to get 'Date' column
df['Date'] = df.index


# Display the stock data
st.write(df)

# Create an instance of BSE
bse = BSE(update_codes=True)

# Get the top gainers
gainers = bse.topGainers()

# Get the top losers
losers = bse.topLosers()

# Display the top gainers and top losers
st.subheader("Top Gainers")
for gainer in gainers:
    st.write(gainer)

    st.subheader("Top Losers")
    for loser in losers:
        st.write(loser)


# create moving average and close
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# visualization

st.header("Graph Of Close Price, MA100 & MA200")

fig = go.Figure()

# Plot the close price
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines',
    name='Close',
    line=dict(color='blue')
))

# Plot the rolling average (ma100)
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=ma100,
    mode='lines',
    name='Rolling Average 100',
    line=dict(color='red')
))

# Plot the rolling average (ma200)
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=ma200,
    mode='lines',
    name='Rolling Average 200',
    line=dict(color='green')
))

fig.update_layout(
    title='Close vs Rolling Averages (200 & 100)',
    xaxis_title='Date',
    yaxis_title='Price',
    showlegend=True
)

st.plotly_chart(fig)

# create moving average terms

df['MA100'] = df['Close'].rolling(window=100, min_periods=0).mean()
df['MA200'] = df['Close'].rolling(window=200, min_periods=0).mean()

# create subplots

fig = make_subplots(rows=2,cols=1 , shared_xaxes=True,
                   vertical_spacing=0.1, subplot_titles=('HDFC','Volume'),
                   row_width=[0.2,0.7])

# Make update visual of graph

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.1, subplot_titles=('HDFC', 'Volume'))

fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)

fig.add_trace(go.Scatter(x=df['Date'], y=df['MA100'], marker_color='grey', name='MA100'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], marker_color='lightgrey', name='MA200'), row=1, col=1)

fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color='red', name='Volume'), row=2, col=1)

fig.update_layout(
    title='HDFC Historical Price Chart',
    xaxis_tickfont_size=12,
    yaxis=dict(
        title='Price ($/Share)',
        titlefont_size=14,
        tickfont_size=12
    ),
    autosize=False,
    width=800,
    height=500,
    margin=dict(l=50, r=200, b=100, t=100, pad=5),  # Adjust the right margin value to create space for legends
    paper_bgcolor='LightSteelBlue',
    legend=dict(x=1.2, y=1.2, traceorder='normal')  # Set the legend position and trace order to right side
)
fig.update_xaxes(rangeslider_visible=False)



st.plotly_chart(fig)


# Splitting the data into training and testing data i.e 70 train & 30 test

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Convert data into Scaler & Standard format

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# load the model

model = load_model("Stock_model")

# testing part

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

# splitting data into tset

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

X_test , y_test = np.array(X_test) , np.array(y_test)

# predict the model

y_predict = model.predict(X_test)

scaler = scaler.scale_
scaler_factor = 1/0.01330318
y_predict = y_predict*scaler_factor
y_test = y_test*scaler_factor

# predict visualization

st.subheader("Original Price V/s Prediction Price")
scaler_factor = 1/0.01330318
y_predict = y_predict * scaler_factor
y_test = y_test * scaler_factor

fig = go.Figure()

# Plot the original price
fig.add_trace(go.Scatter(
    x=df.index,
    y=y_test.flatten(),
    mode='lines',
    name='Original Price',
    line=dict(color='blue')
))

# Plot the predicted price
fig.add_trace(go.Scatter(
    x=df.index,
    y=y_predict.flatten(),
    mode='lines',
    name='Predicted Price',
    line=dict(color='green')
))

fig.update_layout(
    title='Original vs Predicted Price',
    xaxis_title='Time',
    yaxis_title='Price',
    showlegend=True
)

st.plotly_chart(fig)








