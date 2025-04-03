from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class VolatilityAPITest(TestCase):
    """Integration tests for the volatility analysis API endpoint."""

    def setUp(self):
        """Set up test environment."""
        # URL for the volatility endpoint
        self.url_pattern = reverse('volatility_analysis', kwargs={'base': 'USD', 'target': 'EUR'})

        # Create test dataframe for mocking
        self.test_df, self.test_metadata = self._create_test_data()

    def _create_test_data(self, days=30, volatility_level='normal'):
        """Helper to create test data for mocking."""
        # Create date range
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Sort chronologically

        # Create base price data
        base_price = 1.0

        # Create price data with different volatility patterns
        if volatility_level == 'high':
            # High volatility - larger random changes
            np.random.seed(42)
            changes = np.random.normal(0, 0.02, days)
        elif volatility_level == 'low':
            # Low volatility - smaller random changes
            np.random.seed(42)
            changes = np.random.normal(0, 0.001, days)
        else:  # normal
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

        # Create metadata
        metadata = {
            'data_source': 'Alpha Vantage',
            'dataset_type': 'Forex Daily',
            'dataset_id': f"forex_daily_test_{datetime.now().strftime('%Y%m%d')}",
            'time_object': {
                'timestamp': datetime.now().isoformat(),
                'timezone': 'GMT+0'
            }
        }

        return df, metadata

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_forex_daily')
    def test_volatility_analysis_success(self, mock_get_forex_daily):
        """Test successful volatility analysis request."""
        # Set up mock
        mock_get_forex_daily.return_value = (self.test_df, self.test_metadata)

        # Make request
        response = self.client.get(self.url_pattern)

        # Check response
        self.assertEqual(response.status_code, 200)

        # Check response format (ADAGE 3.0)
        data = response.json()
        self.assertEqual(data['data_source'], 'Alpha Vantage')
        self.assertEqual(data['dataset_type'], 'Currency Volatility Analysis')
        self.assertIn('dataset_id', data)
        self.assertIn('time_object', data)
        self.assertIn('events', data)

        # Check event structure
        self.assertEqual(len(data['events']), 1)
        event = data['events'][0]
        self.assertEqual(event['event_type'], 'volatility_analysis')

        # Check attributes
        attrs = event['attributes']
        self.assertEqual(attrs['base_currency'], 'USD')
        self.assertEqual(attrs['target_currency'], 'EUR')
        self.assertIn('current_volatility', attrs)
        self.assertIn('average_volatility', attrs)
        self.assertIn('volatility_level', attrs)
        self.assertIn('trend', attrs)
        self.assertIn('confidence_score', attrs)

        # Check cache headers are set
        self.assertIn('Cache-Control', response)

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_forex_daily')
    def test_volatility_analysis_with_days_parameter(self, mock_get_forex_daily):
        """Test volatility analysis with days parameter."""
        # Set up mock
        mock_get_forex_daily.return_value = (self.test_df, self.test_metadata)

        # Make request with days parameter within allowed range
        response = self.client.get(f"{self.url_pattern}?days=60")

        # Check response
        self.assertEqual(response.status_code, 200)

        # Verify days parameter is used
        data = response.json()
        event = data['events'][0]
        self.assertEqual(event['time_object']['duration'], 60)
        self.assertEqual(event['attributes']['analysis_period_days'], 60)

    @patch('myapp.Service.volatilityService.VolatilityService.calculate_volatility')
    def test_volatility_analysis_with_different_volatility(self, mock_calculate_volatility):
        """Test volatility analysis with different volatility levels."""
        # Since we've tested the volatility calculation in unit tests, here we just verify
        # the API can process and return different volatility levels correctly

        # Set up mock for normal volatility (default)
        mock_calculate_volatility.return_value = {
            'data_source': 'Alpha Vantage',
            'dataset_type': 'Currency Volatility Analysis',
            'dataset_id': 'test_volatility_id',
            'time_object': {
                'timestamp': '2023-04-01T12:00:00.000Z',
                'timezone': 'GMT+0'
            },
            'events': [
                {
                    'time_object': {
                        'timestamp': '2023-04-01T12:00:00.000Z',
                        'duration': 30,
                        'duration_unit': 'day',
                        'timezone': 'GMT+0'
                    },
                    'event_type': 'volatility_analysis',
                    'attributes': {
                        'base_currency': 'USD',
                        'target_currency': 'EUR',
                        'current_volatility': 9.5,
                        'average_volatility': 8.7,
                        'volatility_level': 'NORMAL',
                        'trend': 'STABLE',
                        'data_points': 30,
                        'confidence_score': 100.0,
                        'analysis_period_days': 30
                    }
                }
            ]
        }

        # Make request
        response = self.client.get(self.url_pattern)

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify the attributes
        self.assertEqual(data['events'][0]['attributes']['volatility_level'], 'NORMAL')

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_forex_daily')
    def test_volatility_analysis_no_data(self, mock_get_forex_daily):
        """Test error handling when no data is available."""
        # Set up mock to raise ValueError
        mock_get_forex_daily.side_effect = ValueError("No data available for USD/EUR in the last 30 days")

        # Make request
        response = self.client.get(self.url_pattern)

        # Check response - should now be 404 Not Found
        self.assertEqual(response.status_code, 404)

        # Check error response format (ADAGE 3.0)
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error']['type'], 'NoDataError')
        self.assertIn('No data available', data['error']['message'])

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_forex_daily')
    def test_volatility_analysis_server_error(self, mock_get_forex_daily):
        """Test error handling for server errors."""
        # Set up mock to raise generic exception with API error message
        mock_get_forex_daily.side_effect = Exception("Invalid API request: Service unavailable")

        # Make request
        response = self.client.get(self.url_pattern)

        # Check response - should now be 503 Service Unavailable
        self.assertEqual(response.status_code, 503)

        # Check error response format (ADAGE 3.0)
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error']['type'], 'ExternalServiceError')
        self.assertIn('external data provider', data['error']['message'])

        # Test with different type of server error
        mock_get_forex_daily.side_effect = Exception("Internal server error")

        # Make request
        response = self.client.get(self.url_pattern)

        # Check response - should still be 500 Internal Server Error
        self.assertEqual(response.status_code, 500)

        # Check error response
        data = response.json()
        self.assertEqual(data['error']['type'], 'ServerError')

    def test_volatility_analysis_invalid_currency(self):
        """Test with invalid currency codes."""
        # Test with invalid base currency
        url_invalid_base = reverse('volatility_analysis', kwargs={'base': 'INVALID', 'target': 'EUR'})
        response_invalid_base = self.client.get(url_invalid_base)

        # Should return 400 Bad Request with our improved error handling
        self.assertEqual(response_invalid_base.status_code, 400)
        data = response_invalid_base.json()
        self.assertIn('error', data)
        self.assertEqual(data['error']['type'], 'InvalidCurrencyError')
        self.assertIn('Invalid base currency code', data['error']['message'])

        # Test with invalid target currency
        url_invalid_target = reverse('volatility_analysis', kwargs={'base': 'USD', 'target': 'INVALID'})
        response_invalid_target = self.client.get(url_invalid_target)

        # Should return 400 Bad Request with our improved error handling
        self.assertEqual(response_invalid_target.status_code, 400)
        data = response_invalid_target.json()
        self.assertIn('error', data)
        self.assertEqual(data['error']['type'], 'InvalidCurrencyError')
        self.assertIn('Invalid target currency code', data['error']['message'])

    def test_volatility_analysis_with_invalid_params(self):
        """Test parameter validation for other error cases."""
        # Test when days parameter is not a valid integer
        response_invalid_days = self.client.get(f"{self.url_pattern}?days=not_a_number")
        self.assertEqual(response_invalid_days.status_code, 400)
        data = response_invalid_days.json()
        self.assertEqual(data['error']['type'], 'ValidationError')
        self.assertIn('must be an integer value', data['error']['message'])

        # Test days below minimum (should now return 400 instead of silently correcting)
        response_small = self.client.get(f"{self.url_pattern}?days=3")
        self.assertEqual(response_small.status_code, 400)
        data = response_small.json()
        self.assertEqual(data['error']['type'], 'ValidationError')
        self.assertIn('must be at least 7', data['error']['message'])

        # Test days above maximum (should now return 400 instead of silently correcting)
        response_large = self.client.get(f"{self.url_pattern}?days=500")
        self.assertEqual(response_large.status_code, 400)
        data = response_large.json()
        self.assertEqual(data['error']['type'], 'ValidationError')
        self.assertIn('must be at most 365', data['error']['message'])

    def test_volatility_analysis_same_currencies(self):
        """Test with same base and target currencies."""
        # Test with same currency for base and target
        url_same = reverse('volatility_analysis', kwargs={'base': 'USD', 'target': 'USD'})
        response_same = self.client.get(url_same)

        # Should return 400 Bad Request
        self.assertEqual(response_same.status_code, 400)
        data = response_same.json()
        self.assertEqual(data['error']['type'], 'InvalidInputError')
        self.assertIn('must be different', data['error']['message'])
