import requests
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import generics
from django.utils.dateparse import parse_datetime
from myapp.Models.financialNewsModel import FinancialNews
from myapp.Serializers.financialNewsSerializer import FinancialNewsSerializer
from datetime import datetime, timedelta

# External API URL (Example: NewsAPI)
NEWS_API_URL = "https://newsapi.org/v2/everything"
API_KEY = "05e994f176cb4adf80b524a7fb2d00c8"  # Replace with your actual API key


class FetchFinancialNewsView(APIView):

    def get(self, request, *args, **kwargs):
        today = datetime.today()
        one_month_ago = today - timedelta(days=28)
        from_date = one_month_ago.strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        params = {
            "q": "forex OR exchange rate OR currency volatility",
            "from": from_date,
            "to": to_date,
            "sortBy": "relevancy",
            "language": "en",
            "domains": (
                "forbes.com,bloomberg.com,wsj.com,ft.com,reuters.com,marketwatch.com,cnbc.com,"
                "investing.com,nytimes.com,bbc.com,msn.com,news.yahoo.com,cnn.com,independent.co.uk,"
                "guardian.co.uk,wsj.com,abcnews.go.com,ft.com"
            ),
            "pageSize": 100,  # Limit to 100 results
            "apiKey": API_KEY,
        }

        page = 1
        total_articles = []

        # Fetch articles in pages until we have 100 articles or no more articles
        while len(total_articles) < 100:
            params["page"] = page
            response = requests.get(NEWS_API_URL, params=params)

            if response.status_code == 200:
                articles = response.json()["articles"]
                total_articles.extend(articles)

                if len(articles) < 100:  # If less than 100 results, stop paginating
                    break
                page += 1
            else:
                return Response(
                    {"error": f"Failed to fetch news: {response.status_code}, {response.json()}"},
                    status=response.status_code
                )

        # Store articles in the database if they don't already exist
        stored_news = []
        for article in total_articles[:100]:  # Ensure we only store up to 100 articles
            if not FinancialNews.objects.filter(url=article["url"]).exists():  # Avoid duplicates
                news = FinancialNews.objects.create(
                    title=article["title"],
                    source=article["source"]["name"],
                    date=parse_datetime(article["publishedAt"])  # Convert string to datetime object
                )
                stored_news.append(news)

        return Response(
            {"message": "News data fetched and stored", "news_count": len(stored_news)},
            status=201
        )


class financialNewsListView(generics.ListAPIView):
    queryset = FinancialNews.objects.all().order_by("-date")  # Order by published date
    serializer_class = FinancialNewsSerializer
