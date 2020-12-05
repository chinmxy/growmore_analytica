# Import libraries
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def calculate_mean_sentiment(tickers, n):
    # Declare Output Dictionary
    output = {}
    
    # Get Data
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        resp = urlopen(req)    
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    # Iterate through the news
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text() 
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
                
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]
            
            parsed_news.append([ticker, date, time, text])
      
    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    # View Data 
    news['Date'] = pd.to_datetime(news.Date).dt.date

    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

    news_restructured = news.to_dict('split')['data']
    news_selected_list = []
    mean_sentiment = 0
    for i in range(n):
        print(news_restructured[i])
        print("\n\n")
        news_selected_list.append(news_restructured[i])
    for i in range(n):
        mean_sentiment += news_selected_list[i][7]
    output['mean_sentiment'] = round(mean_sentiment, 2)
    output['news'] = news_selected_list
    # print(output)
    # values = []
    # for ticker in tickers: 
    #     dataframe = news_dict[ticker]
    #     dataframe = dataframe.set_index('Ticker')
    #     dataframe = dataframe.drop(columns = ['Headline'])
    #     mean = round(dataframe['compound'].head(n).mean(), 2)
    #     values.append(mean)
        
    # df = pd.DataFrame(list(zip(tickers, values)), columns =['Ticker', 'Mean Sentiment']) 
    # df = df.set_index('Ticker')
    # df = df.sort_values('Mean Sentiment', ascending=False)
    # print(values)

calculate_mean_sentiment(["AAPL"], 2)