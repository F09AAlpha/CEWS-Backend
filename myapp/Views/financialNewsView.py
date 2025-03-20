from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
from datetime import datetime
from django.utils.dateparse import parse_datetime
from myapp.Models.financialNewsModel import FinancialNews
from myapp.Serializers.financialNewsSerializer import FinancialNewsSerializer
from rest_framework import generics

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = "TNEZTD2EW034X3N"  # Replace with your API key
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class FetchFinancialNewsView(APIView):
    def post(self, request, *args, **kwargs):
        # Extract parameters from the request body
        symbol = request.data.get("symbol", "AAPL")  # Default: AAPL (Apple Inc.)

        # Alpha Vantage API parameters
        params = {
            "function": "TIME_SERIES_DAILY",  # Example: Daily time series
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "compact",  # Use "full" for full historical data
        }

        # Fetch data from Alpha Vantage API
        response = requests.get(ALPHA_VANTAGE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            time_series = data.get("Time Series (Daily)", {})

            if time_series:
                # Store the financial data in the database
                stored_data = []
                for date, values in time_series.items():
                    if not FinancialNews.objects.filter(
                        symbol=symbol,
                        date=parse_datetime(date)  # Use `date` instead of `url`
                    ).exists():
                        # Save the data to the database
                        news = FinancialNews.objects.create(
                            symbol=symbol,
                            date=parse_datetime(date),
                            open_price=float(values["1. open"]),
                            high_price=float(values["2. high"]),
                            low_price=float(values["3. low"]),
                            close_price=float(values["4. close"]),
                            volume=int(values["5. volume"])
                        )
                        stored_data.append(news)

                return Response(
                    {"message": "Financial data fetched and stored", "data_count": len(stored_data)},
                    status=status.HTTP_201_CREATED
                )
            else:
                return Response(
                    {"error": "No financial data found in the API response."},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            return Response(
                {"error": f"Failed to fetch financial data from Alpha Vantage: {response.status_code}"},
                status=response.status_code,
            )


class financialNewsListView(generics.ListAPIView):
    queryset = FinancialNews.objects.all().order_by("-date")  # Order by date
    serializer_class = FinancialNewsSerializer
