from datetime import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
from django.utils.dateparse import parse_datetime
from myapp.Models.currencyNewsModel import CurrencyNewsAlphaV
from myapp.Serializers.currencyNewsSerializer import CurrencyNewsSerializer
from rest_framework import generics



# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = "VCQD8OHRMOLM1H10"  # Replace with your API key
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class FetchCurrencyNewsView(APIView):
    def post(self, request, *args, **kwargs):
        currency = request.data.get("currency", "USD")  # Default: USD

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": currency,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "sort": "LATEST",
            "limit": 10
        }

        response = requests.get(ALPHA_VANTAGE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            news_data = data.get("feed", [])

            if news_data:
                stored_news = []
                for article in news_data:
                    publication_date = parse_datetime(article.get("time_published"))
                    aware_publication_date = publication_date.replace(tzinfo=timezone.utc)

                    if not CurrencyNewsAlphaV.objects.filter(url=article.get("url")).exists():
                        news_data = {
                            "title": article.get("title"),
                            "source": article.get("source"),
                            "url": article.get("url"),
                            "summary": article.get("summary"),
                            "sentiment_score": float(article.get("overall_sentiment_score", 0)),
                            "sentiment_label": article.get("overall_sentiment_label", "neutral"),
                            "publication_date": aware_publication_date,
                            "currency": currency
                        }

                        serializer = CurrencyNewsSerializer(data=news_data)
                        if serializer.is_valid():
                            serializer.save()
                            stored_news.append(serializer.data)

                return Response(
                    {"message": "Currency news data fetched and stored", "news": stored_news},
                    status=status.HTTP_201_CREATED
                )
            else:
                return Response(
                    {"error": "No currency news data found in the API response."},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            return Response(
                {"error": f"Failed to fetch currency news data from Alpha Vantage: {response.status_code}"},
                status=response.status_code,
            )

class CurrencyNewsListView(generics.RetrieveAPIView):
    serializer_class = CurrencyNewsSerializer

    def get_object(self):
        # Get the currency from the URL
        currency = self.kwargs.get('currency')

        # Fetch the latest entry for the specified currency
        try:
            return CurrencyNewsAlphaV.objects.filter(currency=currency).latest('published_at')
        except CurrencyNewsAlphaV.DoesNotExist:
            return None

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance:
            serializer = self.get_serializer(instance)
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(
                {"error": f"No news found for currency: {self.kwargs.get('currency')}"},
                status=status.HTTP_404_NOT_FOUND
            )
