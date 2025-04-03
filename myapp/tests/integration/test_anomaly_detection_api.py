from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch
import json
import pandas as pd
from datetime import datetime, timedelta

from myapp.Models.anomalyDetectionModel import AnomalyDetectionResult


class AnomalyDetectionAPITest(TestCase):
    """Integration tests for the anomaly detection API endpoint."""

    def setUp(self):
        """Set up test environment."""
        # URL for the anomaly detection endpoint
        self.url = reverse('anomaly-detection')

        # Test request data
        self.valid_request_data = {
            'base': 'USD',
            'target': 'EUR',
            'days': 30
        }

        # Create test dataframe for mocking
        self.test_df = self._create_test_dataframe()

    def _create_test_dataframe(self, data_points=30):
        """Helper to create a test dataframe with exchange rate data."""
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(data_points)]
        dates.reverse()  # Sort chronologically

        # Create exchange rate values with a specific pattern
        base_rate = 1.0
        rates = [base_rate + i * 0.01 for i in range(data_points)]

        # Add anomalies only if we have enough data points
        if data_points > 21:  # Only add anomalies if we have enough points
            rates[7] = base_rate + 0.15  # Spike
            rates[21] = base_rate - 0.1   # Drop
        elif data_points > 7:
            rates[7] = base_rate + 0.15  # Just add the first spike

        # Create DataFrame
        data = {
            'date': dates,
            'close': rates
        }
        return pd.DataFrame(data)

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_valid_request(self, mock_get_rates):
        """Test a valid anomaly detection request."""
        # Mock the Alpha Vantage service
        mock_get_rates.return_value = self.test_df

        # Make a POST request to the API
        response = self.client.post(
            self.url,
            data=json.dumps(self.valid_request_data),
            content_type='application/json'
        )

        # Check the response
        self.assertEqual(response.status_code, 200)

        # Parse the response data
        data = response.json()

        # Verify ADAGE 3.0 format structure
        self.assertEqual(data['data_source'], 'Alpha Vantage')
        self.assertEqual(data['dataset_type'], 'Currency Exchange Rates')
        self.assertEqual(data['dataset_id'], "exchange_anomaly_USD_EUR")
        self.assertIn('time_object', data)
        self.assertIn('events', data)

        # Verify events data
        events = data['events']
        self.assertGreater(len(events), 0)

        # Verify structure of an event
        event = events[0]
        self.assertIn('time_object', event)
        self.assertEqual(event['event_type'], 'exchange_rate_anomaly')
        self.assertIn('attribute', event)

        # Verify event attributes
        attr = event['attribute']
        self.assertEqual(attr['base_currency'], 'USD')
        self.assertEqual(attr['target_currency'], 'EUR')
        self.assertIn('rate', attr)
        self.assertIn('z_score', attr)
        self.assertIn('percent_change', attr)

        # Verify that the model is saved to the database
        self.assertEqual(AnomalyDetectionResult.objects.count(), 1)
        result = AnomalyDetectionResult.objects.first()
        self.assertEqual(result.base_currency, 'USD')
        self.assertEqual(result.target_currency, 'EUR')
        self.assertEqual(result.analysis_period_days, 30)

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_invalid_request_missing_fields(self, mock_get_rates):
        """Test request with missing required fields."""
        # Make a POST request with missing target currency
        invalid_data = {'base': 'USD'}
        response = self.client.post(
            self.url,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )

        # Check for bad request response
        self.assertEqual(response.status_code, 400)
        self.assertIn('detail', response.json())

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_invalid_request_same_currencies(self, mock_get_rates):
        """Test request with same base and target currencies."""
        # Make a POST request with same base and target
        invalid_data = {
            'base': 'USD',
            'target': 'USD',
            'days': 30
        }
        response = self.client.post(
            self.url,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )

        # Check for bad request response
        self.assertEqual(response.status_code, 400)
        self.assertIn('detail', response.json())

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_insufficient_data(self, mock_get_rates):
        """Test handling insufficient data for analysis."""
        # Create a small dataframe (insufficient data)
        small_df = self._create_test_dataframe(data_points=5)
        mock_get_rates.return_value = small_df

        # Make a POST request to the API
        response = self.client.post(
            self.url,
            data=json.dumps(self.valid_request_data),
            content_type='application/json'
        )

        # Check for the error response (wrapped in a ProcessingError, so 500 status code)
        self.assertEqual(response.status_code, 500)
        self.assertIn('detail', response.json())

        # Verify message indicates insufficient data
        error_detail = response.json()['detail']
        self.assertIn("Error processing request", error_detail)

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_api_error_handling(self, mock_get_rates):
        """Test handling of API errors."""
        # Mock an API error
        mock_get_rates.side_effect = Exception("API error")

        # Make a POST request to the API
        response = self.client.post(
            self.url,
            data=json.dumps(self.valid_request_data),
            content_type='application/json'
        )

        # Check for internal server error response
        self.assertEqual(response.status_code, 500)
        self.assertIn('detail', response.json())
