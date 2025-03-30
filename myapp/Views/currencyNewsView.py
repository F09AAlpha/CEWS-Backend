from datetime import timezone,datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
from django.utils.dateparse import parse_datetime
from myapp.Models.currencyNewsModel import CurrencyNewsAlphaV
from myapp.Serializers.currencyNewsSerializer import CurrencyNewsSerializer
from rest_framework import generics
import os
import uuid
from myapp.Serializers.currencyNewsSerializer import AdageNewsDatasetSerializer


# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')  # Replace with your API key
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class FetchCurrencyNewsView(APIView):
    def post(self, request, currency, *args, **kwargs):
        # Get the limit parameter from query parameters, default is 10
        limit = request.query_params.get("limit", 10)

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": currency,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "sort": "LATEST",
            "limit": limit
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

                # Create ADAGE 3.0 response format
                current_time = datetime.now(timezone.utc)
                date_str = current_time.strftime("%Y%m%d")
                
                adage_response = {
                    "data_source": "Alpha Vantage",
                    "dataset_type": "currency_news",
                    "dataset_id": f"currency-news-{currency}-{date_str}",
                    "time_object": {
                        "timestamp": current_time.isoformat(),
                        "timezone": "UTC"
                    },
                    "events": []
                }
                
                # No need to create response events since we just stored the data
                # But we provide the ADAGE structure for consistency
                
                adage_serializer = AdageNewsDatasetSerializer(data=adage_response)
                if adage_serializer.is_valid():
                    return Response(
                        adage_serializer.data,
                        status=status.HTTP_200_OK
                    )
                else:
                    return Response(
                        {"message": f"Currency news data fetched and stored for {currency}", "currency": currency},
                        status=status.HTTP_200_OK
                    )
            else:
                return Response(
                    {"message": f"No currency news data found for {currency}"},
                    status=status.HTTP_200_OK,
                )
        else:
            return Response(
                {"detail": f"Failed to fetch currency news data from Alpha Vantage: {response.status_code}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CurrencyNewsListView(generics.ListAPIView):
    serializer_class = CurrencyNewsSerializer
    
    def get_queryset(self):
        # Get query parameters
        currency = self.request.query_params.get('currency')
        sentiment = self.request.query_params.get('sentiment_score')
        limit = self.request.query_params.get('limit', 10)  # Default to 10 if not specified

        # Start with all objects
        queryset = CurrencyNewsAlphaV.objects.all().order_by('-publication_date')

        # Apply filters if provided
        if currency:
            queryset = queryset.filter(currency=currency)
        if sentiment:
            try:
                sentiment_value = float(sentiment)
                # Filter sentiment_score within ±0.05 of the entered value
                queryset = queryset.filter(
                    sentiment_score__gte=sentiment_value - 0.05, sentiment_score__lte=sentiment_value + 0.05
                )
            except ValueError:
                # Handle the case where sentiment is not a valid float
                print(f"Invalid sentiment value: {sentiment}")  # Debugging

        # Debugging: Print the generated SQL query
        print(queryset.query)

        # Limit results
        return queryset[:int(limit)]

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()

        if not queryset:
            return Response([], status=status.HTTP_200_OK)

        # Convert to ADAGE 3.0 format
        current_time = datetime.now(timezone.utc)
        date_str = current_time.strftime("%Y%m%d")
        
        # Get currency from parameters or use 'all' if not specified
        currency = request.query_params.get('currency', 'all')
        
        # Create the ADAGE 3.0 structure
        adage_data = {
            "data_source": "Alpha Vantage",
            "dataset_type": "currency_news",
            "dataset_id": f"currency-news-{currency}-{date_str}",
            "time_object": {
                "timestamp": current_time.isoformat(),
                "timezone": "UTC"
            },
            "events": []
        }
        
        # Create events from queryset
        for news_item in queryset:
            event_id = f"CN-{date_str}-{uuid.uuid4().hex[:8]}"
            
            # Create event data structure
            event = {
                "time_object": {
                    "timestamp": news_item.publication_date.isoformat(),
                    "duration": 0,
                    "duration_unit": "second",
                    "timezone": "UTC"
                },
                "event_type": "currency_news",
                "event_id": event_id,
                "attributes": {
                    "title": news_item.title,
                    "source": news_item.source,
                    "url": news_item.url,
                    "summary": news_item.summary,
                    "sentiment_score": news_item.sentiment_score,
                    "sentiment_label": news_item.sentiment_label,
                    "currency": news_item.currency
                }
            }
            
            adage_data["events"].append(event)
        
        # Validate the structure using the serializer
        adage_serializer = AdageNewsDatasetSerializer(data=adage_data)
        if adage_serializer.is_valid():
            return Response(adage_serializer.data, status=status.HTTP_200_OK)
        else:
            # Fall back to the regular serializer if the ADAGE structure is not valid
            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)