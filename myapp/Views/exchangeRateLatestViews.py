from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# You would typically store this in environment variables
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class CurrencyRateView(APIView):
    """
    API view for retrieving the latest exchange rate between two currencies.
    This is a direct call to external API and not stored in the DB.
    """

    def get(self, request, base, target):
        """
        Handle GET request for latest exchange rate.

        Args:
            request: The HTTP request
            base: Base currency code (e.g., EUR)
            target: Target currency code (e.g., USD)

        Returns:
            Response with the current exchange rate information
        """
        try:
            # Validate currency codes
            base = base.upper()
            target = target.upper()

            if len(base) != 3 or len(target) != 3:
                return Response(
                    {"detail": "Invalid currency code. Currency codes must be 3 characters."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Call Alpha Vantage API
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": base,
                "to_currency": target,
                "apikey": ALPHA_VANTAGE_API_KEY
            }

            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
            response.raise_for_status()  # Raise exception for non-2xx responses

            data = response.json()

            # Check if the response contains error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return Response(
                    {"detail": "External API error: Unable to fetch exchange rate data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Extract exchange rate data
            exchange_rate_data = data.get("Realtime Currency Exchange Rate", {})

            if not exchange_rate_data:
                return Response(
                    {"detail": "No exchange rate data found for the specified currency pair."},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Format response according to our API schema
            rate_response = {
                "base": exchange_rate_data.get("1. From_Currency Code", base),
                "target": exchange_rate_data.get("3. To_Currency Code", target),
                "rate": exchange_rate_data.get("5. Exchange Rate", "0.00000000"),
                "timestamp": exchange_rate_data.get(
                    "6. Last Refreshed", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
                "source": "Alpha Vantage"
            }

            return Response(rate_response)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            return Response(
                {"detail": "Error connecting to external currency data provider."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            logger.error(f"Unexpected error in CurrencyRateView: {str(e)}")
            return Response(
                {"detail": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
