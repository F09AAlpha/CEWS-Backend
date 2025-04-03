import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from myapp.Service.correlationService import CorrelationService
from myapp.Exceptions.exceptions import CorrelationDataUnavailable


class TestCorrelationServiceCalculations(unittest.TestCase):
    """Test cases for the calculation methods in CorrelationService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create the service
        self.service = CorrelationService()

        # Mock the alpha_vantage_service attribute
        self.service.alpha_vantage_service = MagicMock()

        # Create test data
        self.exchange_df = self._create_exchange_dataframe()
        self.news_df = self._create_news_dataframe()
        self.econ_df = self._create_econ_dataframe()

        # Create a properly indexed exchange dataframe for forex_daily mock
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='D')
        self.forex_df = pd.DataFrame({
            'close': np.random.random(100) * 10 + 1,
            'open': np.random.random(100) * 10 + 1,
            'high': np.random.random(100) * 10 + 1.5,
            'low': np.random.random(100) * 10 + 0.5
        }, index=dates)

        self.test_metadata = {
            'data_source': 'Alpha Vantage',
            'dataset_type': 'Forex Daily',
            'dataset_id': 'test_dataset_id',
            'time_object': {
                'timestamp': datetime.now().isoformat(),
                'timezone': 'GMT+0'
            }
        }

    def _create_exchange_dataframe(self, days=30):
        """Helper to create a test exchange rate DataFrame."""
        # Create date range
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).date() for i in range(days)]
        dates.reverse()  # Sort chronologically

        # Create price data with some trend and volatility
        np.random.seed(42)  # For reproducibility

        # Create base price series with trend (slight upward)
        base_price = 1.0
        trend = np.linspace(0, 0.05, days)  # Slight upward trend
        noise = np.random.normal(0, 0.01, days)  # Daily noise
        prices = base_price + trend + np.cumsum(noise)  # Cumulative noise for random walk

        # Calculate returns and add some volatility
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)  # Add 0 for first day

        # Create rolling volatility (20-day window)
        volatility = []
        for i in range(days):
            if i < 20:
                # For first 20 days, use available data
                window = returns[:i+1]
                vol = np.std(window) * np.sqrt(252) * 100 if len(window) > 1 else 5.0
            else:
                # Use 20-day rolling window
                window = returns[i-20:i]
                vol = np.std(window) * np.sqrt(252) * 100
            volatility.append(vol)

        # Create dataframe
        data = {
            'date': dates,
            'close': prices,
            'return': returns,
            'volatility': volatility
        }

        return pd.DataFrame(data)

    def _create_news_dataframe(self, days=30):
        """Helper to create a test news sentiment DataFrame."""
        # Create date range (same as exchange data)
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).date() for i in range(days)]
        dates.reverse()  # Sort chronologically

        # Create test currencies
        currencies = ['USD', 'EUR']

        # Create sentiment data with some correlation to price movements
        np.random.seed(42)

        news_data = []
        for date in dates:
            for currency in currencies:
                # Create sentiment metrics with currency as column prefix rather than as rows
                sentiment_score_mean = np.random.normal(0.2 if currency == 'USD' else 0.1, 0.3)
                sentiment_score_std = abs(np.random.normal(0.2, 0.1))
                sentiment_score_count = max(1, int(np.random.normal(10, 3)))

                news_data.append({
                    'date': date,
                    'currency': currency,
                    'sentiment_score_mean': sentiment_score_mean,
                    'sentiment_score_std': sentiment_score_std,
                    'sentiment_score_count': sentiment_score_count
                })

        # Convert to DataFrame
        news_df = pd.DataFrame(news_data)

        # Pivot to match the format expected by the service
        # This creates columns like USD_sentiment_score_mean, EUR_sentiment_score_mean, etc.
        pivoted_news = news_df.pivot(index='date', columns='currency', values=[
            'sentiment_score_mean', 'sentiment_score_std', 'sentiment_score_count'
        ])

        # Flatten the column names
        pivoted_news.columns = [f"{currency}_{col}" for col, currency in pivoted_news.columns]

        # Reset index to get date as a column
        pivoted_news = pivoted_news.reset_index()

        return pivoted_news

    def _create_econ_dataframe(self, days=30):
        """Helper to create a test economic indicators DataFrame."""
        # Create date range (same as exchange data)
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).date() for i in range(days)]
        dates.reverse()  # Sort chronologically

        # Create economic indicators with trends
        np.random.seed(42)

        # Simulate economic indicators
        cpi_base = 2.0
        cpi_trend = np.linspace(0, 0.5, days)
        cpi_noise = np.random.normal(0, 0.1, days)
        cpi = cpi_base + cpi_trend + cpi_noise

        unemployment_base = 5.0
        unemployment_trend = np.linspace(0, -1.0, days)  # Decreasing trend
        unemployment_noise = np.random.normal(0, 0.2, days)
        unemployment = unemployment_base + unemployment_trend + unemployment_noise
        unemployment = np.maximum(unemployment, 3.0)  # Floor at 3%

        ffr_base = 1.75
        ffr_steps = np.zeros(days)
        ffr_steps[10:20] = 0.25  # Rate hike in the middle
        ffr = ffr_base + np.cumsum(ffr_steps)

        treasury_base = 2.5
        treasury_trend = np.linspace(0, 0.3, days)
        treasury_noise = np.random.normal(0, 0.05, days)
        treasury = treasury_base + treasury_trend + treasury_noise

        # Create dataframe
        data = {
            'date': dates,
            'cpi': cpi,
            'unemployment_rate': unemployment,
            'federal_funds_rate': ffr,
            'treasury_yield': treasury
        }

        return pd.DataFrame(data)

    def test_init(self):
        """Test service initialization."""
        service = CorrelationService()
        self.assertIsNotNone(service.alpha_vantage_service)

    def test_merge_datasets(self):
        """Test merging datasets for correlation analysis."""
        # Call the method
        merged_df, data_completeness = self.service.merge_datasets(
            self.exchange_df.copy(), self.news_df.copy(), self.econ_df.copy()
        )

        # Verify result
        self.assertIsInstance(merged_df, pd.DataFrame)
        self.assertGreater(len(merged_df), 0)

        # Check that all essential columns are present
        expected_columns = ['close', 'return', 'volatility']
        for col in expected_columns:
            self.assertIn(col, merged_df.columns)

        # Check for economic indicator columns
        for indicator in ['cpi', 'unemployment_rate', 'federal_funds_rate', 'treasury_yield']:
            self.assertIn(indicator, merged_df.columns)

        # Check data completeness score
        self.assertGreaterEqual(data_completeness, 0.0)
        self.assertLessEqual(data_completeness, 1.0)

    def test_nan_replacement(self):
        """Test replacing NaN values in dictionaries."""
        # Create a dictionary with NaN values
        test_dict = {
            'key1': 1.0,
            'key2': np.nan,
            'key3': 3.0,
            'key4': np.nan
        }

        # Create a deep copy to test
        modified_dict = test_dict.copy()

        # Replace NaN values
        for key, value in list(modified_dict.items()):
            if pd.isna(value) or np.isnan(value):
                modified_dict[key] = 0.0

        # Check that NaN values are replaced
        self.assertEqual(modified_dict['key1'], 1.0)
        self.assertEqual(modified_dict['key2'], 0.0)
        self.assertEqual(modified_dict['key3'], 3.0)
        self.assertEqual(modified_dict['key4'], 0.0)

        # Try to JSON serialize - should succeed
        json_str = json.dumps(modified_dict)
        self.assertTrue(len(json_str) > 0)

    def test_format_adage_response(self):
        """Test formatting correlation results to ADAGE 3.0 format."""
        # Create a mock CorrelationResult object
        correlation_result = MagicMock()
        correlation_result.base_currency = 'USD'
        correlation_result.target_currency = 'EUR'
        correlation_result.lookback_days = 90
        correlation_result.confidence_score = 65.5
        correlation_result.data_completeness = 75.0
        correlation_result.analysis_date = datetime.now()

        correlation_result.exchange_news_correlation = {
            'close_to_USD_sentiment_score_mean': 0.25,
            'close_to_EUR_sentiment_score_mean': 0.15
        }
        correlation_result.exchange_economic_correlation = {
            'close_to_cpi': 0.32,
            'close_to_unemployment_rate': -0.28
        }
        correlation_result.volatility_news_correlation = {
            'volatility_to_USD_sentiment_score_mean': -0.22,
            'volatility_to_EUR_sentiment_score_mean': -0.17
        }
        correlation_result.volatility_economic_correlation = {
            'volatility_to_cpi': 0.15,
            'volatility_to_unemployment_rate': -0.12
        }
        correlation_result.top_influencing_factors = [
            {
                'factor': 'treasury_yield',
                'impact': 'medium',
                'correlation': 0.47,
                'type': 'economic'
            }
        ]

        # Format the response
        response = self.service.format_adage_response(correlation_result)

        # Verify response structure
        self.assertIsInstance(response, dict)
        self.assertEqual(response['data_source'], 'Currency Exchange Warning System')
        self.assertEqual(response['dataset_type'], 'currency_correlation_analysis')
        self.assertIn('dataset_id', response)
        self.assertIn('time_object', response)
        self.assertIn('events', response)

        # Check events structure
        self.assertEqual(len(response['events']), 1)
        event = response['events'][0]
        self.assertEqual(event['event_type'], 'correlation_analysis')

        # Check attributes
        attrs = event['attributes']
        self.assertEqual(attrs['base_currency'], 'USD')
        self.assertEqual(attrs['target_currency'], 'EUR')
        self.assertEqual(attrs['confidence_score'], 65.5)
        self.assertEqual(attrs['data_completeness'], 75.0)
        self.assertEqual(attrs['analysis_period_days'], 90)

        # Check correlation sections
        correlations = attrs['correlations']
        self.assertIn('news_sentiment', correlations)
        self.assertIn('economic_indicators', correlations)
        self.assertIn('volatility_news', correlations)

    @patch('myapp.Models.correlationModel.CorrelationResult.objects')
    def test_get_latest_correlation(self, mock_objects):
        """Test retrieving the latest correlation result."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.base_currency = 'USD'
        mock_result.target_currency = 'EUR'
        mock_result.analysis_date = datetime.now()

        # Configure mock to return our result when filtered and ordered
        mock_filter = MagicMock()
        mock_filter.order_by.return_value.first.return_value = mock_result
        mock_objects.filter.return_value = mock_filter

        # Call the method
        result = self.service.get_latest_correlation('USD', 'EUR')

        # Verify the result
        self.assertEqual(result, mock_result)
        mock_objects.filter.assert_called_with(
            base_currency='USD',
            target_currency='EUR'
        )
        mock_filter.order_by.assert_called_with('-analysis_date')

    @patch('myapp.Models.correlationModel.CorrelationResult.objects')
    def test_get_latest_correlation_no_data(self, mock_objects):
        """Test error handling when no correlation data is available."""
        # Configure mock to return None
        mock_filter = MagicMock()
        mock_filter.order_by.return_value.first.return_value = None
        mock_objects.filter.return_value = mock_filter

        # Call the method and expect exception
        with self.assertRaises(CorrelationDataUnavailable):
            self.service.get_latest_correlation('USD', 'EUR')

        # Verify the filter was called with correct args
        mock_objects.filter.assert_called_with(
            base_currency='USD',
            target_currency='EUR'
        )

    @patch('myapp.Models.correlationModel.CorrelationResult.objects')
    def test_get_latest_correlation_exception(self, mock_objects):
        """Test error handling when database query fails."""
        # Configure mock to raise an exception
        mock_objects.filter.side_effect = Exception("Database error")

        # Call the method and expect our custom exception
        with self.assertRaises(CorrelationDataUnavailable):
            self.service.get_latest_correlation('USD', 'EUR')

        # Verify the filter was called
        mock_objects.filter.assert_called_with(
            base_currency='USD',
            target_currency='EUR'
        )

    @patch('myapp.Service.correlationService.CorrelationResult')
    def test_analyze_and_store_correlations(self, mock_result_class):
        """Test the complete correlation analysis workflow with simpler mocking."""
        # Create a mock service with controlled behavior
        mock_service = MagicMock(spec=CorrelationService)

        # Mock the analyze_and_store_correlations method to call our internal mock implementation
        real_method = CorrelationService.analyze_and_store_correlations
        mock_service.analyze_and_store_correlations = lambda *args, **kwargs: real_method(mock_service, *args, **kwargs)

        # Set up alpha_vantage_service mock
        mock_service.alpha_vantage_service = MagicMock()
        mock_service.alpha_vantage_service.get_forex_daily.return_value = (self.forex_df, self.test_metadata)

        # Mock internal methods
        mock_service.get_news_sentiment_data.return_value = self.news_df
        mock_service.get_economic_indicator_data.return_value = self.econ_df
        mock_service.merge_datasets.return_value = (pd.concat([self.forex_df.reset_index(), self.news_df], axis=1), 0.85)

        # Mock calculate_correlations
        mock_service.calculate_correlations.return_value = ({
            'exchange_news_correlation': {'close_to_USD_sentiment_score_mean': 0.35},
            'exchange_economic_correlation': {'close_to_cpi': 0.42},
            'volatility_news_correlation': {'volatility_to_USD_sentiment_score_std': -0.28},
            'volatility_economic_correlation': {'volatility_to_unemployment_rate': -0.15},
            'top_influencing_factors': [
                {'factor': 'cpi', 'impact': 'medium', 'correlation': 0.42, 'type': 'economic'}
            ],
            'confidence_score': 68.5
        }, 0.42)

        # Mock CorrelationResult
        mock_result = MagicMock()
        mock_result_class.return_value = mock_result

        # Mock transaction.atomic to be a no-op context manager
        with patch('django.db.transaction.atomic') as mock_atomic:
            mock_atomic.return_value.__enter__.return_value = None
            mock_atomic.return_value.__exit__.return_value = None

            # Call the method
            result = mock_service.analyze_and_store_correlations('USD', 'EUR', 90)

            # Verify it was called correctly
            mock_service.alpha_vantage_service.get_forex_daily.assert_called_with('USD', 'EUR')
            mock_service.get_news_sentiment_data.assert_called()
            mock_service.get_economic_indicator_data.assert_called()
            mock_result.save.assert_called_once()

            # Verify result is correct
            self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main()
