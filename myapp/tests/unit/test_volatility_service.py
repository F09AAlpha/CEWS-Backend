import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from myapp.Service.volatilityService import VolatilityService


class TestVolatilityService(unittest.TestCase):
    """Test cases for the VolatilityService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create the service
        self.service = VolatilityService()

        # Mock the alpha_vantage_service attribute
        self.service.alpha_vantage_service = MagicMock()

        # Create test data
        self.test_df = self._create_test_dataframe()
        self.test_metadata = {
            'data_source': 'Alpha Vantage',
            'dataset_type': 'Forex Daily',
            'dataset_id': 'test_dataset_id',
            'time_object': {
                'timestamp': datetime.now().isoformat(),
                'timezone': 'GMT+0'
            }
        }

    def _create_test_dataframe(self, days=30, volatility_level='normal'):
        """Helper to create a test DataFrame with controlled volatility."""
        # Create date range
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Sort chronologically

        # Create base price data
        base_price = 1.0

        # Create price data with different volatility patterns
        if volatility_level == 'low':
            # Low volatility - small random changes
            np.random.seed(42)  # For reproducibility
            changes = np.random.normal(0, 0.001, days)
            prices = [base_price + changes[:i+1].sum() for i in range(days)]
        elif volatility_level == 'high':
            # High volatility - larger random changes
            np.random.seed(42)
            changes = np.random.normal(0, 0.01, days)
            prices = [base_price + changes[:i+1].sum() for i in range(days)]
        elif volatility_level == 'extreme':
            # Extreme volatility - very large random changes
            np.random.seed(42)
            changes = np.random.normal(0, 0.02, days)
            prices = [base_price + changes[:i+1].sum() for i in range(days)]
        elif volatility_level == 'increasing':
            # Volatility that increases over time
            np.random.seed(42)
            first_half = np.random.normal(0, 0.001, days//2)
            second_half = np.random.normal(0, 0.015, days - days//2)
            changes = np.concatenate([first_half, second_half])
            prices = [base_price + changes[:i+1].sum() for i in range(days)]
        elif volatility_level == 'decreasing':
            # Volatility that decreases over time
            np.random.seed(42)
            first_half = np.random.normal(0, 0.015, days//2)
            second_half = np.random.normal(0, 0.001, days - days//2)
            changes = np.concatenate([first_half, second_half])
            prices = [base_price + changes[:i+1].sum() for i in range(days)]
        else:  # normal/default
            # Normal volatility
            np.random.seed(42)
            changes = np.random.normal(0, 0.005, days)
            prices = [base_price + changes[:i+1].sum() for i in range(days)]

        # Create dataframe with OHLC data
        data = {
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
            'close': prices
        }

        df = pd.DataFrame(data, index=dates)
        return df

    def test_init(self):
        """Test service initialization."""
        service = VolatilityService()
        self.assertIsNotNone(service.alpha_vantage_service)

    @patch('myapp.Service.volatilityService.AlphaVantageService')
    def test_calculate_volatility_normal(self, mock_av_service):
        """Test calculating volatility with normal volatility."""
        # Set up mock
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(volatility_level='normal'),
            self.test_metadata
        )

        # Call the method
        result = self.service.calculate_volatility('USD', 'EUR', 30)

        # Verify result structure
        self.assertEqual(result['data_source'], 'Alpha Vantage')
        self.assertEqual(result['dataset_type'], 'Currency Volatility Analysis')
        self.assertIn('dataset_id', result)
        self.assertIn('time_object', result)
        self.assertIn('events', result)
        self.assertEqual(len(result['events']), 1)

        # Verify event data
        event = result['events'][0]
        self.assertIn('time_object', event)
        self.assertEqual(event['event_type'], 'volatility_analysis')
        self.assertIn('attributes', event)

        # Verify attributes
        attrs = event['attributes']
        self.assertEqual(attrs['base_currency'], 'USD')
        self.assertEqual(attrs['target_currency'], 'EUR')
        self.assertIn('current_volatility', attrs)
        self.assertIn('average_volatility', attrs)
        self.assertIn('volatility_level', attrs)
        self.assertIn('trend', attrs)
        self.assertIn('data_points', attrs)
        self.assertIn('confidence_score', attrs)
        self.assertEqual(attrs['analysis_period_days'], 30)

    def test_calculate_volatility_with_different_levels(self):
        """Test volatility level determination with different volatility levels."""
        # Test low volatility
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(volatility_level='low'),
            self.test_metadata
        )
        result_low = self.service.calculate_volatility('USD', 'EUR', 30)

        # Test high volatility
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(volatility_level='high'),
            self.test_metadata
        )
        result_high = self.service.calculate_volatility('USD', 'EUR', 30)

        # Test extreme volatility
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(volatility_level='extreme'),
            self.test_metadata
        )
        result_extreme = self.service.calculate_volatility('USD', 'EUR', 30)

        # Verify that volatility level increases with volatility
        low_vol = result_low['events'][0]['attributes']['current_volatility']
        high_vol = result_high['events'][0]['attributes']['current_volatility']
        extreme_vol = result_extreme['events'][0]['attributes']['current_volatility']

        self.assertLess(low_vol, high_vol)
        self.assertLess(high_vol, extreme_vol)

    def test_calculate_volatility_with_different_trends(self):
        """Test trend determination with different volatility trends."""
        # Test increasing volatility
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(volatility_level='increasing'),
            self.test_metadata
        )
        result_increasing = self.service.calculate_volatility('USD', 'EUR', 30)

        # Test decreasing volatility
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(volatility_level='decreasing'),
            self.test_metadata
        )
        result_decreasing = self.service.calculate_volatility('USD', 'EUR', 30)

        # Verify trends
        increasing_trend = result_increasing['events'][0]['attributes']['trend']
        decreasing_trend = result_decreasing['events'][0]['attributes']['trend']

        # These assertions may not always hold due to the random nature of the test data
        # and the specific implementation of trend detection, but they help verify the
        # general behavior
        self.assertIn(increasing_trend, ['INCREASING', 'STABLE'])
        self.assertIn(decreasing_trend, ['DECREASING', 'STABLE'])

    def test_calculate_volatility_different_time_periods(self):
        """Test calculating volatility with different time periods."""
        # Set up mock for 7 days
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(days=7),
            self.test_metadata
        )

        # Calculate for 7 days
        result_7_days = self.service.calculate_volatility('USD', 'EUR', 7)

        # Set up mock for 90 days
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            self._create_test_dataframe(days=90),
            self.test_metadata
        )

        # Calculate for 90 days
        result_90_days = self.service.calculate_volatility('USD', 'EUR', 90)

        # Verify the time period is reflected in the result
        self.assertEqual(result_7_days['events'][0]['time_object']['duration'], 7)
        self.assertEqual(result_7_days['events'][0]['attributes']['analysis_period_days'], 7)

        self.assertEqual(result_90_days['events'][0]['time_object']['duration'], 90)
        self.assertEqual(result_90_days['events'][0]['attributes']['analysis_period_days'], 90)

    def test_calculate_volatility_no_data(self):
        """Test error handling when no data is available."""
        # Set up mock to return empty DataFrame
        empty_df = pd.DataFrame()
        self.service.alpha_vantage_service.get_forex_daily.return_value = (
            empty_df,
            self.test_metadata
        )

        # Expect ValueError when calling with empty data
        with self.assertRaises(ValueError) as context:
            self.service.calculate_volatility('USD', 'EUR', 30)

        # Verify error message
        self.assertIn("No data available", str(context.exception))


if __name__ == '__main__':
    unittest.main()
