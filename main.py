import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def load_tweet_dataset(path_dataset):
    df_list = []
    for split in ["train_stockemo","val_stockemo","test_stockemo"]: ## We are joining three datasets (training, validation and test)
        path = path_dataset / f"{split}.csv"
        df = pd.read_csv(path)
        df = df[['date','ticker','original','senti_label']]
        df.rename(columns={'original':'text', 'senti_label':'sentiment'}, inplace=True)
        df["data_type"] = split # We trace the dataset type
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def load_price_dataset(path_dataset, tickers):
    price_dir = Path(path_dataset)
    price_dfs = {}
    for ticker in tickers:
        file = price_dir / f"{ticker.replace('.','-')}.csv"
        dfp = pd.read_csv(file, parse_dates=['Date'])
        dfp['date'] = dfp['Date'].dt.date
        dfp.set_index('date', inplace=True)
        price_dfs[ticker] = dfp[['Open','Close','Adj Close', 'High', 'Low', 'Volume']]
    return price_dfs

def plot_tweet_histogram(df):
    df_tweet = df.copy()
    # NOTE: A "week" interval is the best INTERVAL to group the data??? I dont know. We need to find the best interval.
    df_tweet['week'] = pd.to_datetime(df_tweet['date']).dt.to_period('W').apply(lambda r: r.start_time)
    # Group by week and sentiment
    weekly_sentiment_counts = df_tweet.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
    # Ensure both columns exist
    if 'bullish' not in weekly_sentiment_counts.columns:
        weekly_sentiment_counts['bullish'] = 0
    if 'bearish' not in weekly_sentiment_counts.columns:
        weekly_sentiment_counts['bearish'] = 0
    weekly_sentiment_counts = weekly_sentiment_counts[['bullish', 'bearish']]
    weeks = weekly_sentiment_counts.index
    x = np.arange(len(weeks))
    plt.figure(figsize=(14, 6))
    plt.bar(x, weekly_sentiment_counts['bullish'], label='Bullish', color='green', alpha=0.7)
    plt.bar(x, weekly_sentiment_counts['bearish'], bottom=weekly_sentiment_counts['bullish'], label='Bearish', color='red', alpha=0.7)
    plt.xlabel('Week')
    plt.ylabel('Number of Tweets')
    plt.title(f"Weekly Tweet Distribution by Sentiment for {df_tweet['ticker'].unique()[0]}")
    plt.xticks(x, [str(w.date()) for w in weeks], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_price_data(price_df, df_tweet):
    # cols = [['Open','Close','Adj Close', 'High', 'Low'],['Volume']]
    cols = [['Open','Close'],['Volume']]
    colors = [['blue','orange'],['red']]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    # Plot all but the last on the left, last on the right
    left_cols = cols[0]
    right_col = cols[1][0]
    # ax .. Plot OPEN and CLOSE plus Tweets
    price_df[left_cols].plot(ax=ax,color=colors[0],linestyle='-')
    xs = df_tweet['date']
    ys = np.repeat(price_df['Close'].mean(),len(xs))
    ax.plot(xs,ys,marker='x',color='olive',linestyle='',markersize=4,alpha=0.5,label='New Tweets')
    
    ax.set_ylabel(', '.join(left_cols))
    ax.legend(left_cols+['New Tweets'], loc='upper left')
    # ax2 .. Plot VOLUME
    ax2 = ax.twinx()
    price_df[right_col].plot(ax=ax2, color=colors[1],linestyle='--',linewidth=0.5)
    ax2.set_ylabel(right_col)
    ax2.legend([right_col], loc='upper right')
    plt.title(f"Price and Tweets for {ticker}")
    plt.tight_layout()
    plt.show()

def plot_both_tweets_distribution_and_price(df_tweet, price_df):
    ## This is my try to plot the price and the tweets distribution together
    ## I guess that there are better ways to do this (in financial analysis)
    ## Investigate the best way to do this
    cols = [['Open', 'Close']]
    colors = [['blue', 'orange']]
    df = df_tweet.copy()
    # Group tweets by week
    df['week'] = pd.to_datetime(df['date']).dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sentiment_counts = df.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
    if 'bullish' not in weekly_sentiment_counts.columns:
        weekly_sentiment_counts['bullish'] = 0
    if 'bearish' not in weekly_sentiment_counts.columns:
        weekly_sentiment_counts['bearish'] = 0
    weekly_sentiment_counts = weekly_sentiment_counts[['bullish', 'bearish']]
    weeks = weekly_sentiment_counts.index
    # Get the Close price for each week (use last available Close in week)
    price_df = price_df.copy()
    price_df['week'] = pd.to_datetime(price_df.index).to_period('W').to_timestamp()
    weekly_close = price_df.groupby('week')['Close'].last()
    # Prepare x positions (week start dates)
    x = np.array([w for w in weeks])
    # Prepare y positions (top of Close for each week)
    y = weekly_close.reindex(weeks).values
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # Plot price data
    price_df[cols[0]].plot(ax=ax, color=colors[0], linestyle='-')
    ax.set_ylabel(', '.join(cols[0]))
    # Overlay stacked bars at the top of Close for each week
    bar_width = 5  # days, for visual width
    bullish = weekly_sentiment_counts['bullish'].values
    bearish = weekly_sentiment_counts['bearish'].values
    # Convert x (week start) to matplotlib date numbers
    x_dates = [pd.Timestamp(w) for w in weeks]
    x_nums = [mdates.date2num(d) for d in x_dates]
    # Plot bullish bars
    ax.bar(x_dates, bullish, width=bar_width, bottom=y, color='green', alpha=0.7, label='Bullish (tweets)')
    # Plot bearish bars stacked on bullish
    ax.bar(x_dates, bearish, width=bar_width, bottom=y+bullish, color='red', alpha=0.7, label='Bearish (tweets)')
    # Legend and title
    ax.legend(loc='upper left')
    ax.set_title(f"Price and Weekly Tweet Sentiment for {df['ticker'].unique()[0]}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_tweet = Path("dataset/StockEmotions/tweet")
    base_price = Path("dataset/StockEmotions/price")
    df_tweet = load_tweet_dataset(base_tweet)
    tickers = df_tweet['ticker'].unique()
    price_dfs = load_price_dataset(base_price,tickers = tickers)
    # print(df_tweet.head())
    # print(price_dfs[tickers[0]].head())

    ticker = "TSLA"
    plot_tweet_histogram(df_tweet[df_tweet['ticker']==ticker])
    plot_price_data(price_dfs[ticker],df_tweet[df_tweet['ticker']==ticker])
    plot_both_tweets_distribution_and_price(df_tweet[df_tweet['ticker']==ticker],price_dfs[ticker])