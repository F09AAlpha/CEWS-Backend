import requests
import pandas as pd
from django.conf import settings
from datetime import datetime, timedelta
from rest_framework.exceptions import NotFound
import logging
import os


# Custom exception classes
class AlphaVantageError(Exception):
    """Base exception for Alpha Vantage API errors"""
    pass


class RateLimitError(AlphaVantageError):
    """Raised when API rate limit is exceeded"""
    pass


class TemporaryAPIError(AlphaVantageError):
    """Raised for temporary API issues like network errors"""
    pass


class InvalidRequestError(AlphaVantageError):
    """Raised for invalid API requests"""
    pass


class AlphaVantageService:
    BASE_URL = "https://www.alphavantage.co/query"

    # Added logger for better diagnostics
    logger = logging.getLogger(__name__)

    def __init__(self, api_key=None):
        # Try to get API key directly from environment first, then fall back to settings
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY') or settings.ALPHA_VANTAGE_API_KEY
        # Log first few chars for debugging
        self.logger.info(f"Initialized Alpha Vantage service with API key: {self.api_key[:4]}...")

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
        This uses the Time Series FX (Daily) endpoint to get real historical data.

        Args:
            from_currency: Base currency code
            to_currency: Target currency code
            days: Number of days of data to return

        Returns:
            DataFrame with exchange rate data
        """
        try:
            # Call the Time Series FX (Daily) endpoint for real historical data
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'outputsize': 'full',  # Get full data set (up to 20 years)
                'apikey': self.api_key
            }

            self.logger.info(f"Fetching historical exchange rate data for {from_currency}/{to_currency}")
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
                raise InvalidRequestError(f"Invalid API request: {data['Error Message']}")

            if 'Note' in data and 'API call frequency' in data['Note']:
                raise RateLimitError(f"Alpha Vantage API rate limit: {data['Note']}")

            if 'Time Series FX (Daily)' not in data:
                raise NotFound(f"No historical data found for {from_currency}/{to_currency}")

            # Parse the time series data
            time_series = data['Time Series FX (Daily)']

            # Create DataFrame from the time series data
            df = pd.DataFrame.from_dict(time_series, orient='index')

            # Convert column names from API format to our standard format
            df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close'
            }, inplace=True)

            # Convert values to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)

            # Convert index to datetime and sort
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Filter to requested time period
            start_date = datetime.now().date() - timedelta(days=days)
            filtered_df = df[df.index.date >= start_date].copy()

            # Add metadata as attributes
            filtered_df.attrs['base_currency'] = from_currency
            filtered_df.attrs['target_currency'] = to_currency

            # Reset index to make date a column
            filtered_df.reset_index(inplace=True)
            filtered_df.rename(columns={'index': 'date'}, inplace=True)

            self.logger.info(f"Retrieved {len(filtered_df)} days of historical data for {from_currency}/{to_currency}")
            return filtered_df

        except (AlphaVantageError, Exception) as e:
            self.logger.error(f"Error retrieving historical exchange rate data: {str(e)}")
            raise

    def get_forex_daily(self, from_currency, to_currency, outputsize='full'):
        """
        Fetch daily forex rates for a currency pair using the FX_DAILY endpoint.

        Args:
            from_currency: Base currency code
            to_currency: Target currency code
            outputsize: 'full' or 'compact' data set

        Returns:
            tuple: (DataFrame with forex data, metadata dict)
        """
        try:
            # Call the Time Series FX (Daily) endpoint for real historical data
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'outputsize': outputsize,
                'apikey': self.api_key
            }

            self.logger.info(f"Fetching forex daily data for {from_currency}/{to_currency}")
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
                raise InvalidRequestError(f"Invalid API request: {data['Error Message']}")

            if 'Note' in data and 'API call frequency' in data['Note']:
                raise RateLimitError(f"Alpha Vantage API rate limit: {data['Note']}")

            if 'Time Series FX (Daily)' not in data:
                raise NotFound(f"No forex daily data found for {from_currency}/{to_currency}")

            # Extract metadata
            metadata = {
                'data_source': 'Alpha Vantage (FX_DAILY)',
                'last_refreshed': data['Meta Data'].get('5. Last Refreshed', 'Unknown'),
                'time_zone': data['Meta Data'].get('6. Time Zone', 'GMT'),
                'information': data['Meta Data'].get('1. Information', 'FX Daily'),
                'from_symbol': data['Meta Data'].get('2. From Symbol', from_currency),
                'to_symbol': data['Meta Data'].get('3. To Symbol', to_currency)
            }

            # Parse the time series data
            time_series = data['Time Series FX (Daily)']

            # Create DataFrame from the time series data
            df = pd.DataFrame.from_dict(time_series, orient='index')

            # Convert column names from API format to our standard format
            df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close'
            }, inplace=True)

            # Convert values to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)

            # Convert index to datetime and sort
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Add attributes for currency info
            df.attrs['base_currency'] = from_currency
            df.attrs['target_currency'] = to_currency

            self.logger.info(f"Retrieved {len(df)} days of forex data for {from_currency}/{to_currency}")
            return df, metadata

        except (AlphaVantageError, Exception) as e:
            self.logger.error(f"Error retrieving forex daily data: {str(e)}")
            # We don't generate mock data anymore - pass the error up the stack
            raise

    def _get_realtime_exchange_rate(self, from_currency, to_currency):
        """
        Get the real-time exchange rate using CURRENCY_EXCHANGE_RATE endpoint
        which is available in the free tier.

        Returns:
            tuple: (exchange_rate, metadata)
        """
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_currency,
            'to_currency': to_currency,
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
                raise InvalidRequestError(f"Invalid API request: {data['Error Message']}")

            if 'Note' in data and 'API call frequency' in data['Note']:
                raise RateLimitError(f"Alpha Vantage API rate limit: {data['Note']}")

            # Extract the exchange rate
            if 'Realtime Currency Exchange Rate' not in data:
                raise NotFound(f"No exchange rate data found for {from_currency}/{to_currency}")

            exchange_rate = float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])

            # Extract metadata
            metadata = {
                'data_source': 'Alpha Vantage (Real-time)',
                'last_refreshed': data['Realtime Currency Exchange Rate']['6. Last Refreshed'],
                'timezone': data['Realtime Currency Exchange Rate']['7. Time Zone']
            }

            self.logger.info(f"Retrieved real-time exchange rate: {from_currency}/{to_currency} = {exchange_rate}")

            return exchange_rate, metadata

        except Exception as e:
            self.logger.error(f"Error retrieving real-time exchange rate: {str(e)}")
            raise

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
