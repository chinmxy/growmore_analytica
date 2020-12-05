from newsapi.newsapi_client import NewsApiClient

# Init
news_api = NewsApiClient(api_key='cc0446450bcc4e46a91abd02e33d5f85')

# /v2/top-headlines
top_headlines = news_api.get_top_headlines(q='bitcoin')

# print(top_headlines)

all_articles = news_api.get_everything(q='microsoft',
                                      from_param='2020-11-15',
                                      to='2017-12-02',
                                      language='en',
                                      sort_by='relevancy')

print(all_articles['articles'][1])
print(len(all_articles['articles']))