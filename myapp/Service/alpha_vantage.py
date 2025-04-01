import requests
import pandas as pd
from django.conf import settings
from datetime import datetime, timedelta
from rest_framework.exceptions import NotFound


# Custom exception classes
class AlphaVantageError(Exception):
    """Base exception for Alpha Vantage API errors"""
    pass


class RateLimitError(AlphaVantageError):
    """Raised when API rate limit is exceeded"""
    pass


class InvalidRequestError(AlphaVantageError):
    """Raised when request parameters are invalid"""
    pass


class TemporaryAPIError(AlphaVantageError):
    """Raised when Alpha Vantage service is temporarily unavailable"""
    pass


class AlphaVantageService:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key=None):
        self.api_key = api_key or settings.ALPHA_VANTAGE_API_KEY

    def get_exchange_rates(self, from_currency, to_currency, days=30):
        """
        Get exchange rates for the specified number of days.
        This is a wrapper around get_forex_daily to match the interface expected by AnomalyDetectionService.

        Args:
            from_currency: Base currency code
            to_currency: Target currency code
            days: Number of days of data to return

        Returns:
            DataFrame with exchange rate data
        """
        # Get the full dataset
        df, _ = self.get_forex_daily(from_currency, to_currency)

        # Filter to requested time period
        start_date = datetime.now().date() - timedelta(days=days)
        filtered_df = df[df.index.date >= start_date].copy()

        # Reset index to make date a column
        filtered_df.reset_index(inplace=True)
        filtered_df.rename(columns={'index': 'date', 'close': 'close'}, inplace=True)

        return filtered_df

    def get_forex_daily(self, from_currency, to_currency, outputsize='full'):
        """
        Fetch daily forex rates for a currency pair
        """
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'outputsize': outputsize,
            'apikey': self.api_key
        }

        try:
            response = requests.get(self.BASE_URL, params=params)

            # Handle HTTP errors
            if response.status_code == 429:
                raise RateLimitError("Alpha Vantage API rate limit exceeded")
            elif response.status_code >= 500:
                raise TemporaryAPIError(f"Alpha Vantage server error: {response.status_code}")
            elif response.status_code != 200:
                raise AlphaVantageError(f"Alpha Vantage API error: {response.status_code} - {response.text}")

            data = response.json()

            # Check for API error messages
            if 'Error Message' in data:
                if 'Invalid API call' in data['Error Message']:
                    raise InvalidRequestError(f"Invalid API request: {data['Error Message']}")
                raise AlphaVantageError(f"Alpha Vantage API error: {data['Error Message']}")

            if 'Note' in data and 'API call frequency' in data['Note']:
                raise RateLimitError(f"Alpha Vantage API rate limit: {data['Note']}")

            # Extract time series data
            time_series_key = 'Time Series FX (Daily)'
            if time_series_key not in data:
                raise NotFound(f"No time series data found for {from_currency}/{to_currency}")

            # Extract metadata for ADAGE 3.0 FORMAT
            metadata = {
                'data_source': 'Alpha Vantage',
                'dataset_type': 'Forex Daily',
                'dataset_id': f"forex_daily_{from_currency}_{to_currency}_{datetime.now().strftime('%Y%m%d')}",
                'time_object': {
                    'timestamp': datetime.now().isoformat(),
                    'timezone': 'GMT+0'
                }
            }

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')

            # Rename columns
            df.columns = [col.split('. ')[1] for col in df.columns]

            # Convert string values to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            # Add date as column
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()  # Sort by date

            return df, metadata

        except requests.exceptions.RequestException as e:
            raise TemporaryAPIError(f"Network error accessing Alpha Vantage API: {str(e)}")
