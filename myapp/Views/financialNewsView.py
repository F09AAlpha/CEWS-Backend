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
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class FetchFinancialNewsView(APIView):
    def post(self, request, *args, **kwargs):
        """Fetch financial news data from Alpha Vantage API and store it."""
        symbol = request.data.get("symbol", "AAPL")  # Default to AAPL if no symbol is provided

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "sort": "LATEST",
            "limit": 10,
        }

        try:
            response = requests.get(ALPHA_VANTAGE_URL, params=params)
            response.raise_for_status()  # Raise an error for non-200 responses
        except requests.exceptions.RequestException as e:
            return Response(
                {"error": f"Failed to fetch financial news data: {str(e)}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        data = response.json()
        news_data = data.get("feed", [])

        # Return 204 No Content if there is no news data
        if not news_data:
            return Response(
                {"message": "No financial news available."},
                status=status.HTTP_204_NO_CONTENT,
            )

        stored_news = []
        current_time = datetime.now(timezone.utc)
        date_str = current_time.strftime("%Y%m%d")
        is_new_data = False  # Track if any new data is stored

        for article in news_data:
            publication_date = parse_datetime(article.get("time_published"))
            aware_publication_date = (
                publication_date.replace(tzinfo=timezone.utc)
                if publication_date
                else datetime.now(timezone.utc)
            )

            # Check if the article already exists in the database
            if not FinancialNewsAlphaV.objects.filter(url=article.get("url")).exists():
                news_entry = {
                    "title": article.get("title"),
                    "source": article.get("source"),
                    "url": article.get("url"),
                    "summary": article.get("summary"),
                    "sentiment_score": float(article.get("overall_sentiment_score", 0)),
                    "sentiment_label": article.get("overall_sentiment_label", "neutral"),
                    "publication_date": aware_publication_date,
                    "symbol": symbol,
                }

                serializer = FinancialNewsSerializer(data=news_entry)
                if serializer.is_valid():
                    serializer.save()
                    stored_news.append(serializer.data)
                    is_new_data = True  # Mark that new data was stored

        # Construct response in ADAGE format
        adage_response = {
            "data_source": "Alpha Vantage",
            "dataset_type": "financial_news",
            "dataset_id": f"financial-news-{symbol}-{date_str}",
            "time_object": {"timestamp": current_time.isoformat(), "timezone": "UTC"},
            "events": [],
        }

        adage_serializer = AdageFinancialNewsDatasetSerializer(data=adage_response)
        if adage_serializer.is_valid():
            return Response(adage_serializer.data, status=status.HTTP_200_OK)

        # Return appropriate status based on whether new data was stored
        if is_new_data:
            return Response(
                {"message": "Financial news data stored.", "news": stored_news},
                status=status.HTTP_201_CREATED,  # 201 Created for new data
            )
        else:
            return Response(
                {"message": "No new articles stored (duplicates detected)."},
                status=status.HTTP_200_OK,  # 200 OK if all articles were duplicates
            )
