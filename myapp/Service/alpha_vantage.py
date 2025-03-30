import requests
import pandas as pd
from django.conf import settings
from datetime import datetime


class AlphaVantageService:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key=None):
        self.api_key = api_key or settings.ALPHA_VANTAGE_API_KEY

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

        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            raise Exception(f"Alpha Vantage API error: {response.status_code} - {response.text}")

        data = response.json()

        # Check for error messages
        if 'Error Message' in data:
            raise Exception(f"Alpha Vantage API error: {data['Error Message']}")

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

        # Extract time series data
        time_series_key = 'Time Series FX (Daily)'
        if time_series_key not in data:
            raise Exception(f"No time series data found for {from_currency}/{to_currency}")

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
