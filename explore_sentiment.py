import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse
import os

def load_tweet_dataset(path_dataset):
    df_list = []
    for split in ["train_stockemo","val_stockemo","test_stockemo"]:
        path = path_dataset / f"{split}.csv"
        df = pd.read_csv(path)
        df = df[['date','ticker','original','senti_label']]
        df.rename(columns={'original':'text', 'senti_label':'sentiment'}, inplace=True)
        df["data_type"] = split
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

def plot_tweet_histogram(df, ticker, save=False):
    df_tweet = df.copy()
    df_tweet['week'] = pd.to_datetime(df_tweet['date']).dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sentiment_counts = df_tweet.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
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
    plt.title(f"Weekly Tweet Distribution by Sentiment for {ticker}")
    plt.xticks(x, [str(w.date()) for w in weeks], rotation=45)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'output/{ticker}_histogram.png')
        plt.close()
    else:
        plt.show()

def plot_price_data(price_df, df_tweet, ticker, save=False):
    cols = [['Open','Close'],['Volume']]
    colors = [['blue','orange'],['red']]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    left_cols = cols[0]
    right_col = cols[1][0]
    price_df[left_cols].plot(ax=ax,color=colors[0],linestyle='-')
    xs = df_tweet['date']
    ys = np.repeat(price_df['Close'].mean(),len(xs))
    ax.plot(xs,ys,marker='x',color='olive',linestyle='',markersize=4,alpha=0.5,label='New Tweets')
    ax.set_ylabel(', '.join(left_cols))
    ax.legend(left_cols+['New Tweets'], loc='upper left')
    ax2 = ax.twinx()
    price_df[right_col].plot(ax=ax2, color=colors[1],linestyle='--',linewidth=0.5)
    ax2.set_ylabel(right_col)
    ax2.legend([right_col], loc='upper right')
    plt.title(f"Price and Tweets for {ticker}")
    plt.tight_layout()
    if save:
        plt.savefig(f'output/{ticker}_price.png')
        plt.close()
    else:
        plt.show()

def plot_both_tweets_distribution_and_price(df_tweet, price_df, ticker, save=False):
    cols = [['Open', 'Close']]
    colors = [['blue', 'orange']]
    df = df_tweet.copy()
    df['week'] = pd.to_datetime(df['date']).dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sentiment_counts = df.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
    if 'bullish' not in weekly_sentiment_counts.columns:
        weekly_sentiment_counts['bullish'] = 0
    if 'bearish' not in weekly_sentiment_counts.columns:
        weekly_sentiment_counts['bearish'] = 0
    weekly_sentiment_counts = weekly_sentiment_counts[['bullish', 'bearish']]
    weeks = weekly_sentiment_counts.index
    price_df = price_df.copy()
    price_df['week'] = pd.to_datetime(price_df.index).to_period('W').to_timestamp()
    weekly_close = price_df.groupby('week')['Close'].last()
    x = np.array([w for w in weeks])
    y = weekly_close.reindex(weeks).values
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    price_df[cols[0]].plot(ax=ax, color=colors[0], linestyle='-')
    ax.set_ylabel(', '.join(cols[0]))
    bar_width = 5
    bullish = weekly_sentiment_counts['bullish'].values
    bearish = weekly_sentiment_counts['bearish'].values
    x_dates = [pd.Timestamp(w) for w in weeks]
    ax.bar(x_dates, bullish, width=bar_width, bottom=y, color='green', alpha=0.7, label='Bullish (tweets)')
    ax.bar(x_dates, bearish, width=bar_width, bottom=y+bullish, color='red', alpha=0.7, label='Bearish (tweets)')
    ax.legend(loc='upper left')
    ax.set_title(f"Price and Weekly Tweet Sentiment for {ticker}")
    plt.tight_layout()
    if save:
        plt.savefig(f'output/{ticker}_combined.png')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--show', action='store_true', help='Display the plots instead of saving them')
    args = parser.parse_args()

    ticker = args.ticker
    show = args.show

    os.makedirs("output", exist_ok=True)

    base_tweet = Path("dataset/StockEmotions/tweet")
    base_price = Path("dataset/StockEmotions/price")
    df_tweet = load_tweet_dataset(base_tweet)
    tickers = df_tweet['ticker'].unique()
    price_dfs = load_price_dataset(base_price, tickers)

    if ticker not in tickers:
        raise ValueError(f"Ticker '{ticker}' not found in tweet dataset.")

    df_filtered = df_tweet[df_tweet['ticker'] == ticker]
    price_df = price_dfs[ticker]

    plot_tweet_histogram(df_filtered, ticker, save=not show)
    plot_price_data(price_df, df_filtered, ticker, save=not show)
    plot_both_tweets_distribution_and_price(df_filtered, price_df, ticker, save=not show)
