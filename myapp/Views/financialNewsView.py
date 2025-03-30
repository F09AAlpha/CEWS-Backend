from datetime import timezone, datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
import os
from django.utils.dateparse import parse_datetime
from myapp.Models.financialNewsModel import FinancialNewsAlphaV
from myapp.Serializers.financialNewsSerializer import (
    FinancialNewsSerializer,
    AdageFinancialNewsDatasetSerializer,
)


# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class FetchFinancialNewsView(APIView):
    def post(self, request, *args, **kwargs):
        symbol = request.data.get("symbol", "AAPL")  

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "sort": "LATEST",
            "limit": 10
        }

        response = requests.get(ALPHA_VANTAGE_URL, params=params)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            news_data = data.get("feed", [])

            if news_data:
                stored_news = []
                current_time = datetime.now(timezone.utc)
                date_str = current_time.strftime("%Y%m%d")

                for article in news_data:
                    publication_date = parse_datetime(article.get("time_published"))
                    aware_publication_date = publication_date.replace(tzinfo=timezone.utc)

                    if not FinancialNewsAlphaV.objects.filter(url=article.get("url")).exists():
                        news_data = {
                            "title": article.get("title"),
                            "source": article.get("source"),
                            "url": article.get("url"),
                            "summary": article.get("summary"),
                            "sentiment_score": float(article.get("overall_sentiment_score", 0)),
                            "sentiment_label": article.get("overall_sentiment_label", "neutral"),
                            "publication_date": aware_publication_date,
                            "symbol": symbol
                        }

                        serializer = FinancialNewsSerializer(data=news_data)
                        if serializer.is_valid():
                            serializer.save()
                            stored_news.append(serializer.data)

                # Create ADAGE 3.0 response format
                current_time = datetime.now(timezone.utc)
                date_str = current_time.strftime("%Y%m%d")

                adage_response = {
                    "data_source": "Alpha Vantage",
                    "dataset_type": "financial_news",
                    "dataset_id": f"financial-news-{symbol}-{date_str}",
                    "time_object": {
                        "timestamp": current_time.isoformat(),
                        "timezone": "UTC"
                    },
                    "events": []
                }

                adage_serializer = AdageFinancialNewsDatasetSerializer(data=adage_response)
                if adage_serializer.is_valid():
                    return Response(
                        adage_serializer.data,
                        status=status.HTTP_200_OK
                    )
                else:
                    # Fall back to original response format if ADAGE structure validation fails
                    return Response(
                        {"message": "Financial news data fetched and stored", "news": stored_news},
                        status=status.HTTP_201_CREATED
                    )
            else:
                return Response(
                    {"error": "No financial news data found in the API response."},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            return Response(
                {"error": f"Failed to fetch financial news data from Alpha Vantage: {response.status_code}"},
                status=response.status_code,
            )
