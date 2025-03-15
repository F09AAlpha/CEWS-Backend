import requests
from rest_framework.response import Response
from rest_framework.views import APIView
from myapp.Models.financialNewsModel import FinancialNews
from rest_framework import generics
from myapp.Serializers.financialNewsSerializer import FinancialNewsSerializer
from django.utils.dateparse import parse_datetime

# External API URL (Example: NewsAPI)
NEWS_API_URL = "https://newsapi.org/v2/everything"
API_KEY = "05e994f176cb4adf80b524a7fb2d00c8"  # Replace with your actual API key

class FetchFinancialNewsView(APIView):
    def get(self, request, *args, **kwargs):
        params = {
            "q": "forex OR exchange rate OR currency volatility",
            "from": "2025-02-15",
            "to": "2025-03-14",
            "sortBy": "relevancy",
            "language": "en",
            "domains": "forbes.com,bloomberg.com,wsj.com,ft.com,reuters.com,marketwatch.com,cnbc.com,investing.com,nytimes.com,bbc.com,msn.com,news.yahoo.com,cnn.com,independent.co.uk,guardian.co.uk,wsj.com,abcnews.go.com,ft.com",
            "pageSize": 100,  # Limit to 100 results
            "apiKey": API_KEY
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
                return Response({"error": f"Failed to fetch news: {response.status_code}, {response.json()}"}, status=response.status_code)
        
        # Now store the articles into the database if they don't already exist
        stored_news = []
        for article in total_articles[:100]:  # Ensure we only store up to 100 articles
            if not FinancialNews.objects.filter(url=article["url"]).exists():  # Avoid duplicates
                # Create new financial news entry if not already in DB
                news = FinancialNews.objects.create(
                    title=article["title"],
                    source=article["source"]["name"],
                    url=article["url"],
                    published_at=parse_datetime(article["publishedAt"])  # Convert string to datetime object
                )
                stored_news.append(news)
        
        # Return a success message with the count of the stored news articles
        return Response({"message": "News data fetched and stored", "news_count": len(stored_news)}, status=201)

class FinancialNewsListView(generics.ListAPIView):
    queryset = FinancialNews.objects.all().order_by("-published_at")  # Order by published date
    serializer_class = FinancialNewsSerializer  # Ensure the serializer is correctly defined
