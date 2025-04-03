import requests
import pandas as pd
from django.conf import settings
from datetime import datetime, timedelta
from rest_framework.exceptions import NotFound
import logging


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

    # Added logger for better diagnostics
    logger = logging.getLogger(__name__)

    def __init__(self, api_key=None):
        self.api_key = api_key or settings.ALPHA_VANTAGE_API_KEY

    # Currency to region mapping for more accurate economic data correlation
    CURRENCY_REGION_MAP = {
        'USD': 'US',
        'EUR': 'EU',
        'GBP': 'UK',
        'JPY': 'JP',
        'AUD': 'AU',
        'CAD': 'CA',
        'CHF': 'CH',
        'NZD': 'NZ',
        'CNY': 'CN',
        'INR': 'IN',
        'BRL': 'BR',
        'MXN': 'MX',
        'RUB': 'RU',
        'ZAR': 'ZA'
    }

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
            self.logger.info(f"Fetching FX_DAILY data for {from_currency}/{to_currency}")
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

    def get_news_sentiment(self, currencies, from_date, limit=500):
        """
        Get news sentiment data for specific currencies from Alpha Vantage NEWS_SENTIMENT endpoint.

        Args:
            currencies (list): List of currency codes to get sentiment for
            from_date (datetime): Start date for news data
            limit (int): Maximum number of items to return

        Returns:
            pandas.DataFrame: Processed news sentiment data
        """
        self.logger.info(f"Fetching news sentiment for currencies: {currencies}")

        # Convert currencies to topics for Alpha Vantage API
        topics = []
        for currency in currencies:
            # Add the currency itself as a topic in multiple formats
            topics.append(currency)
            topics.append(f"FOREX:{currency}")  # Add FOREX format

            # Add relevant economic topics based on currency region
            region = self.CURRENCY_REGION_MAP.get(currency, '')
            if region:
                topics.append(f"{region}_ECONOMY")
                topics.append(f"{region}_FINANCIAL")
                topics.append(f"{region}_FOREX")  # Add FOREX regional topic

        # Deduplicate topics
        topics = list(set(topics))

        # Join topics with commas for the API
        topics_str = ','.join(topics)

        # Format from_date for the API (time_from parameter)
        # Request a wider time range to increase chances of finding news
        time_from = (from_date - timedelta(days=7)).strftime('%Y%m%dT0000')  # 7 days earlier

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': topics_str,  # Use tickers parameter for currencies and topics
            'time_from': time_from,
            'sort': 'RELEVANCE',
            'limit': min(limit, 1000),  # API has a max of 1000
            'apikey': self.api_key
        }

        try:
            self.logger.debug(f"Requesting news with params: {params}")
            response = requests.get(self.BASE_URL, params=params)

            # Handle HTTP errors
            if response.status_code == 429:
                self.logger.warning("Alpha Vantage API rate limit exceeded for news sentiment")
                return pd.DataFrame()  # Return empty DataFrame
            elif response.status_code != 200:
                self.logger.error(f"Alpha Vantage API error: {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            # Check if we have feed data
            if 'feed' not in data or not data['feed']:
                self.logger.warning(f"No news data found for {topics_str}")
                return pd.DataFrame()

            # Process the news feed into a DataFrame
            news_items = []
            for item in data['feed']:
                # Extract publication date
                pub_date = datetime.strptime(item['time_published'], '%Y%m%dT%H%M%S')

                # Process sentiment data for each currency
                for currency in currencies:
                    # Check ticker sentiment
                    sentiment_found = False

                    # Look for the currency in ticker_sentiment
                    for ticker in item.get('ticker_sentiment', []):
                        ticker_symbol = ticker.get('ticker', '')
                        # Match both "USD" and "FOREX:USD" formats
                        if ticker_symbol == currency or ticker_symbol == f"FOREX:{currency}":
                            sentiment_found = True
                            news_items.append({
                                'date': pub_date.date(),
                                'currency': currency,
                                'sentiment_score': float(ticker.get('ticker_sentiment_score', 0)),
                                'sentiment_label': ticker.get('ticker_sentiment_label', 'Neutral'),
                                'relevance_score': float(ticker.get('relevance_score', 0)),
                                'title': item.get('title', ''),
                                'source': item.get('source', '')
                            })

                    # If no specific sentiment for this currency, use overall sentiment
                    # but with reduced relevance
                    if not sentiment_found and 'overall_sentiment_score' in item:
                        news_items.append({
                            'date': pub_date.date(),
                            'currency': currency,
                            'sentiment_score': float(item.get('overall_sentiment_score', 0)) * 0.7,  # Reduced weight
                            'sentiment_label': item.get('overall_sentiment_label', 'Neutral'),
                            'relevance_score': 0.3,  # Low relevance since it's not currency-specific
                            'title': item.get('title', ''),
                            'source': item.get('source', '')
                        })

            if not news_items:
                self.logger.warning(f"No sentiment data found for currencies: {currencies}")
                return pd.DataFrame()

            return pd.DataFrame(news_items)

        except Exception as e:
            self.logger.error(f"Error fetching news sentiment: {str(e)}")
            return pd.DataFrame()

    def get_economic_indicators(self, currencies, from_date):
        """
        Get economic indicators relevant to the specific currencies.

        Args:
            currencies (list): List of currency codes to get economic data for
            from_date (datetime): Start date for economic data

        Returns:
            pandas.DataFrame: Processed economic indicator data
        """
        self.logger.info(f"Fetching economic indicators for currencies: {currencies}")

        # Get regions relevant to the currencies
        regions = [self.CURRENCY_REGION_MAP.get(currency, '') for currency in currencies]
        regions = [r for r in regions if r]  # Remove empty regions

        if not regions:
            self.logger.warning(f"No recognized regions for currencies: {currencies}")
            return pd.DataFrame()

        # Initialize empty lists for each indicator
        all_data = []

        # Map of indicators to fetch based on which currency is most relevant
        indicator_functions = {
            'US': ['CPI', 'UNEMPLOYMENT', 'FEDERAL_FUNDS_RATE', 'TREASURY_YIELD'],
            'EU': ['CPI', 'UNEMPLOYMENT'],
            'UK': ['CPI', 'UNEMPLOYMENT'],
            'JP': ['CPI', 'UNEMPLOYMENT'],
            # Other regions would typically have fewer indicators available
            'DEFAULT': ['CPI']  # Fallback
        }

        # Track the dates for later alignment
        dates = set()

        # Fetch indicators for each currency region
        for region in regions:
            # Get the list of indicators for this region
            indicators = indicator_functions.get(region, indicator_functions['DEFAULT'])

            for indicator in indicators:
                try:
                    params = {
                        'function': indicator,
                        'apikey': self.api_key
                    }

                    response = requests.get(self.BASE_URL, params=params)

                    if response.status_code != 200:
                        self.logger.warning(f"Failed to get {indicator} data for {region}: {response.status_code}")
                        continue

                    data = response.json()

                    if 'data' not in data:
                        self.logger.warning(f"No data returned for {indicator}")
                        continue

                    for item in data['data']:
                        date = datetime.strptime(item['date'], '%Y-%m-%d').date()

                        # Skip data before from_date
                        if date < from_date.date():
                            continue

                        dates.add(date)

                        # Add region prefix to avoid conflicts between regions
                        indicator_name = f"{region.lower()}_{indicator.lower()}"

                        all_data.append({
                            'date': date,
                            'indicator': indicator_name,
                            'value': float(item['value']),
                            'region': region
                        })

                except Exception as e:
                    self.logger.error(f"Error fetching {indicator} for {region}: {str(e)}")

        if not all_data:
            self.logger.warning("No economic data found")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Pivot the data to get one row per date with columns for each indicator
        pivot_df = df.pivot_table(
            index='date',
            columns='indicator',
            values='value',
            aggfunc='mean'  # In case of duplicates
        ).reset_index()

        return pivot_df
