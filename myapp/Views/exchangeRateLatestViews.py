from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import pytz
import uuid
from myapp.Serializers.exchangeRateLatestSerializer import (
    AdageCurrencyRateSerializer,
    TimeObjectSerializer,
    CurrencyRateAttributesSerializer,
    CurrencyRateEventSerializer
)

# Load environment variables from the .env file
load_dotenv()

logger = logging.getLogger(__name__)


ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class CurrencyRateView(APIView):
    """
    API view for retrieving the latest exchange rate between two currencies.
    This is a direct call to external API and not stored in the DB.
    Returns data in ADAGE 3.0 Data Model format.
    """

    def get(self, request, base, target):
        """
        Handle GET request for latest exchange rate, formatted according to ADAGE 3.0 Data Model.

        Args:
            request: The HTTP request
            base: Base currency code (e.g., EUR)
            target: Target currency code (e.g., USD)

        Returns:
            Response with the exchange rate information formatted according to ADAGE 3.0 Data Model
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
            response.raise_for_status()

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

            # Get current time in UTC
            now = datetime.now(pytz.UTC)
            now_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")

            # Get the timestamp from Alpha Vantage or use current time
            timestamp = exchange_rate_data.get(
                "6. Last Refreshed",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # Create a unique event ID for this rate retrieval
            event_id = f"CE-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

            # Create dataset time object
            dataset_time_object = {
                "timestamp": now_str,
                "timezone": "UTC"
            }

            # Validate time object with serializer
            time_object_serializer = TimeObjectSerializer(data=dataset_time_object)
            if not time_object_serializer.is_valid():
                logger.error(f"Invalid time object format: {time_object_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Create event time object
            event_time_object = {
                "timestamp": timestamp,
                "duration": 0,
                "duration_unit": "second",
                "timezone": "UTC"
            }

            # Create event attributes
            try:
                rate = float(exchange_rate_data.get("5. Exchange Rate", "0.00"))
            except ValueError:
                rate = 0.0
                logger.warning("Could not convert rate to float, using default 0.0")

            attributes = {
                "base": exchange_rate_data.get("1. From_Currency Code", base),
                "target": exchange_rate_data.get("3. To_Currency Code", target),
                "rate": rate,
                "bid_price": exchange_rate_data.get("8. Bid Price", None),
                "ask_price": exchange_rate_data.get("9. Ask Price", None),
                "source": "Alpha Vantage"
            }

            # Validate attributes with serializer
            attributes_serializer = CurrencyRateAttributesSerializer(data=attributes)
            if not attributes_serializer.is_valid():
                logger.error(f"Invalid attributes format: {attributes_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Create event object
            event = {
                "time_object": event_time_object,
                "event_type": "currency_rate",
                "event_id": event_id,
                "attributes": attributes_serializer.validated_data
            }

            # Validate event with serializer
            event_serializer = CurrencyRateEventSerializer(data=event)
            if not event_serializer.is_valid():
                logger.error(f"Invalid event format: {event_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Create complete ADAGE 3.0 formatted response
            adage_data = {
                "data_source": "Alpha Vantage",
                "dataset_type": "currency_exchange_rate",
                "dataset_id": f"currency-{base}-{target}-{now.strftime('%Y%m%d')}",
                "time_object": time_object_serializer.validated_data,
                "events": [event_serializer.validated_data]
            }

            # Validate the entire structure with the main serializer
            adage_serializer = AdageCurrencyRateSerializer(data=adage_data)
            if not adage_serializer.is_valid():
                logger.error(f"Invalid ADAGE format: {adage_serializer.errors}")
                return Response(
                    {"detail": "Error formatting response data."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Return the validated data
            return Response(adage_serializer.validated_data)

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
