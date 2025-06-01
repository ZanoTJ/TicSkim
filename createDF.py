#before run this, install APIs:
# pip install matplotlib
# pip install yfinance
# pip install seaborn
# pip install scikit-learn
# pip install praw
# pip install pipeline
# pip install transformers
# pip install tensorflow
# pip install torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import pipeline

import os
from datetime import datetime, timedelta

import historical   #custom file

global_ticker = ""
visualize = False

class StockDF:
  def loadDFbyTicker():
    # Get stock data from Yahoo Finance
    stock_data = yf.Ticker(global_ticker)
    df = stock_data.history(period="max")     #we can adjust the term

    if df.shape[0] < 100:   #we need at least 100 rows
      raise ValueError("DataFrame has less than 100 rows.")
    
    df.fillna('?', inplace=True)

    return df
  
  def manageNull(df):   ###need to work on this
    return df
  

class SentimentLoad:
  def __init__(self):
    if not is_file_recent(f'./{global_ticker}_reddit_data.csv'):
      historical.launch(global_ticker)
    self.sentiment_task = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

  def run(self):
    df = self.getSentiments(self.getPosts())
    return self.simplifyDF(df)

  def getPosts(self):
    posts = pd.read_csv('./' + global_ticker + '_reddit_data.csv', na_values=False)
    posts = posts[['Title', 'Post_Date', 'Upvotes']]
    posts = posts.sort_values(by='Post_Date', ascending=True)

    return posts
  
  def getSentiments(self, posts):
    sentiments = posts['Title'].apply(self.sentiment_task)

    sentiment_df = sentiments.to_frame(name="Sentiments")
    sentiment_df['label'] = sentiment_df['Sentiments'].apply(lambda x: x[0]['label'])
    sentiment_df['score'] = sentiment_df['Sentiments'].apply(lambda x: x[0]['score'])
    sentiment_df = sentiment_df.drop('Sentiments', axis=1)

    sentiment_df['sentiment_value'] = sentiment_df.apply(self.simplify_sentiment, axis=1)    #positive == score, negative == -score, neutral -= 0.5
    df = pd.concat([posts, sentiment_df], axis=1)

    if visualize:
      print(df)

    return df

  def simplify_sentiment(self, row):
    sentiment_type = row['label']
    sentiment_value = row['score']

    if sentiment_type == 'negative':
      sentiment_value = -sentiment_value
    elif sentiment_type == 'neutral':
      sentiment_value -= 0.5

    return sentiment_value
  
  def simplifyDF(self, df):
    df = df.drop('Title', axis=1)
    df = df.dropna()    #remove null values (either no date or title)

    #Compress the duplicates by averaging 'Upvotes' and 'sentiment_value' cloumns
    grouped_df = df.groupby('Post_Date').agg({
      'Upvotes': 'mean',  # Calculate the average of Upvotes
      'sentiment_value': 'mean'  # Calculate the average of sentiment scores
    }).reset_index()

    #Weight sentiment_value by Upvotes
    grouped_df['log_weighted_sentiment'] = grouped_df['sentiment_value'] * np.log(grouped_df['Upvotes'])
    grouped_df['weighted_sentiment'] = grouped_df['sentiment_value'] * grouped_df['log_weighted_sentiment']

    new_sentiment_df = grouped_df[['Post_Date', 'log_weighted_sentiment']]
    new_sentiment_df = new_sentiment_df.rename(columns={'Post_Date': 'Date', 'log_weighted_sentiment': 'sentiment'})

    return new_sentiment_df      #Final form of sentiment dataframe


def visualization(df):
  plt.figure(figsize=(15,6))  #size of the graph
  plt.plot(df['Close'])
  plt.title(global_ticker + ' Company stock price')
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.show()

def merge_df(stock_df, sentiment_df):
  new_df = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]   #Drop the unnecessary columns

  sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
  new_df.index = pd.to_datetime(new_df.index).tz_localize(None)

  new_df = new_df.merge(sentiment_df, left_index=True, right_on='Date', how='left')         #how='left': keeping all data of new_df, how='right': merge data that only exist in new_sentiment_df
  new_df['sentiment'] = new_df['sentiment'].fillna(0)         #if there is no sentiment data on the date, fill it with 0

  return new_df

def is_file_recent(file_path):
    #file_path = f'./{global_ticker}_reddit_data.csv'
    limit_days = 5
    f_type = os.path.splitext(file_path)[1]  # Gets '.csv'

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist.")
        return False

    if f_type == '.pkl':
      df = pd.read_pickle(file_path)
    elif f_type == '.csv':
      df = pd.read_csv(file_path)

    last_date = df['Date'].iloc[-1]   # get the most recent date
    # Make sure it's a datetime object
    if not isinstance(last_date, datetime):
      last_date = pd.to_datetime(last_date)
    
    now = datetime.now()

    return (now - last_date) <= timedelta(days=limit_days)

def run(is_visual = False):
  #this is main method
  visualize = is_visual
  global global_ticker
  global_ticker = input("Enter the Ticker of the stock: ").strip().upper()
  
  df_path = f'./{global_ticker}_df.pkl'
  if is_file_recent(df_path):
    df = pd.read_pickle(f'./{global_ticker}_df.pkl')
  else:
    #Load Stock data from Yahoo Finance
    stock_df = StockDF.loadDFbyTicker()

    if stock_df.isnull().any(axis=1).any():   #later
      stock_df = StockDF.manageNull(stock_df)

    if visualize:
      visualization(stock_df)

    #Load Sentiments from communities (if it's not recently generated)
    sentiment_instance = SentimentLoad()
    sentiment_df = sentiment_instance.run()

    #Merge stock df and sentiment df (final format of df)
    df = merge_df(stock_df, sentiment_df)
    df.to_pickle(f"{global_ticker}_df.pkl")

  if visualize:
    print(df)
    print(df[df['sentiment'] > 0.0])    #print where sentiment is valid