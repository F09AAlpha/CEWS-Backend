import os
import requests
import logging
from datetime import datetime
from decimal import Decimal
from django.conf import settings
from myapp.Models.exchangeRateLatestModel import CurrencyEvent

logger = logging.getLogger(__name__)


class AlphaVantageService:
    """Service for interacting with Alpha Vantage API"""

    BASE_URL = 'https://www.alphavantage.co/query'

    @staticmethod
    def get_exchange_rate(base, target):
        """
        Retrieve the latest exchange rate from Alpha Vantage

        Args:
            base (str): Base currency code (e.g., EUR)
            target (str): Target currency code (e.g., USD)

        Returns:
            dict: Exchange rate data if successful

        Raises:
            Exception: If API request fails
        """
        try:
            # Get API key from environment or settings
            api_key = os.environ.get('ALPHA_VANTAGE_API_KEY') or settings.ALPHA_VANTAGE_API_KEY

            # Make request to Alpha Vantage API
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': base,
                'to_currency': target,
                'apikey': api_key
            }

            response = requests.get(AlphaVantageService.BASE_URL, params=params)

            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"Alpha Vantage API returned status code {response.status_code}")
                raise Exception(f"API request failed with status {response.status_code}")

            data = response.json()

            # Check if the response contains the expected data
            if 'Realtime Currency Exchange Rate' not in data:
                error_message = data.get('Error Message', 'Invalid response format from Alpha Vantage API')
                logger.error(f"Alpha Vantage API error: {error_message}")
                raise Exception(error_message)

            # Extract relevant information
            exchange_data = data['Realtime Currency Exchange Rate']

            # Create and save the event
            event_id = CurrencyEvent.generate_event_id(base, target)

            # Parse timestamp
            timestamp_str = exchange_data.get('6. Last Refreshed', '')
            try:
                if ' ' in timestamp_str:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d')
            except ValueError:
                timestamp = datetime.now()

            # Create currency event object
            currency_event = CurrencyEvent(
                event_id=event_id,
                base=base,
                target=target,
                rate=Decimal(exchange_data.get('5. Exchange Rate', 0)),
                timestamp=timestamp,
                source='Alpha Vantage'
            )

            # Save to database
            currency_event.save()

            return currency_event

        except Exception as e:
            logger.exception(f"Error getting exchange rate: {str(e)}")
            raise
