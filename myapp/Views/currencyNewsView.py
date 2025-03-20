from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
from django.utils.dateparse import parse_datetime
from myapp.Models.currencyNewsModel import CurrencyNewsAlpha
from myapp.Serializers.currencyNewsSerializer import CurrencyNewsSerializer
from rest_framework import generics

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = "TNEZTD2EW034X3N"  # Replace with your API key
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class FetchCurrencyNewsView(APIView):
    def post(self, request, *args, **kwargs):
        # Extract parameters from the request body
        from_currency = request.data.get("from_currency", "USD")  # Default: USD
        to_currency = request.data.get("to_currency", "EUR")  # Default: EUR

        # Alpha Vantage API parameters
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }

        # Fetch data from Alpha Vantage API
        response = requests.get(ALPHA_VANTAGE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            exchange_rate_data = data.get("Realtime Currency Exchange Rate", {})

            if exchange_rate_data:
                # Extract relevant data
                exchange_rate = float(exchange_rate_data.get("5. Exchange Rate"))
                last_refreshed = parse_datetime(exchange_rate_data.get("6. Last Refreshed"))

                # Check if the data already exists in the database
                if not CurrencyNewsAlpha.objects.filter(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    last_refreshed=last_refreshed  # Use `last_refreshed` instead of `date`
                ).exists():
                    # Save the data to the database
                    currency_data = CurrencyNewsAlpha.objects.create(
                        from_currency=from_currency,
                        to_currency=to_currency,
                        exchange_rate=exchange_rate,
                        last_refreshed=last_refreshed,
                    )
                    serializer = CurrencyNewsSerializer(currency_data)
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
                else:
                    return Response(
                        {"message": "Data already exists in the database."},
                        status=status.HTTP_200_OK,
                    )
            else:
                return Response(
                    {"error": "No exchange rate data found in the API response."},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            return Response(
                {"error": f"Failed to fetch data from Alpha Vantage: {response.status_code}"},
                status=response.status_code,
            )


class CurrencyNewsListView(generics.ListAPIView):
    queryset = CurrencyNewsAlpha.objects.all().order_by("-last_refreshed")  # Order by date
    serializer_class = CurrencyNewsSerializer
