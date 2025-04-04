import os
import unittest
from decimal import Decimal
from django.test import LiveServerTestCase
from rest_framework.test import APIClient
from unittest.mock import patch

from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert


@unittest.skipIf(os.environ.get('SKIP_E2E', 'False') == 'True', "Skipping E2E tests")
class ExchangeRateE2ETests(LiveServerTestCase):
    """
    End-to-end tests for Exchange Rate API.

    Note: These tests make actual HTTP requests to the local test server,
    but mock external dependencies like the Alpha Vantage API.

    To skip these tests, set the SKIP_E2E environment variable to 'True'.
    """

    def setUp(self):
        self.client = APIClient()
        self.base_url = '/api/v1/currency/rates/'

        # Create a test alert
        self.test_alert = ExchangeRateAlert.objects.create(
            alert_id="E2E-TEST-ALERT-123",
            base="USD",
            target="EUR",
            alert_type="above",
            threshold=Decimal('0.90'),
            email="test@example.com"
        )

    def tearDown(self):
        # Clean up created objects
        ExchangeRateAlert.objects.filter(alert_id="E2E-TEST-ALERT-123").delete()

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_exchange_rate_endpoint(self, mock_requests_get):
        """Test the exchange rate endpoint with a mocked external API"""
        # Mock the Alpha Vantage API response
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Realtime Currency Exchange Rate": {
                "1. From_Currency Code": "USD",
                "2. From_Currency Name": "United States Dollar",
                "3. To_Currency Code": "EUR",
                "4. To_Currency Name": "Euro",
                "5. Exchange Rate": "0.92",
                "6. Last Refreshed": "2023-04-01 12:00:00",
                "7. Time Zone": "UTC",
                "8. Bid Price": "0.91",
                "9. Ask Price": "0.93"
            }
        }
        mock_requests_get.return_value = mock_response

        # Make a request to our API using direct URL
        response = self.client.get(f"{self.base_url}USD/EUR/")

        # Assertions
        self.assertEqual(response.status_code, 200)

        # Check response format and data
        data = response.json()
        self.assertEqual(data['data_source'], "Alpha Vantage")
        self.assertEqual(data['dataset_type'], "currency_exchange_rate")
        self.assertIn('dataset_id', data)
        self.assertIn('time_object', data)
        self.assertIn('events', data)

        # Check event details
        event = data['events'][0]
        self.assertEqual(event['event_type'], "currency_rate")
        self.assertEqual(event['attributes']['base'], "USD")
        self.assertEqual(event['attributes']['target'], "EUR")
        self.assertEqual(event['attributes']['rate'], 0.92)

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_exchange_rate_alert_triggering(self, mock_requests_get):
        """Test that alerts are correctly triggered from the API endpoint"""
        # Mock the Alpha Vantage API response with rate above threshold
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Realtime Currency Exchange Rate": {
                "1. From_Currency Code": "USD",
                "3. To_Currency Code": "EUR",
                "5. Exchange Rate": "0.92",  # Above our threshold of 0.90
                "6. Last Refreshed": "2023-04-01 12:00:00"
            }
        }
        mock_requests_get.return_value = mock_response

        # Make API request using direct URL
        response = self.client.get(f"{self.base_url}USD/EUR/")

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify alert record was deleted after triggering
        alert_exists = ExchangeRateAlert.objects.filter(alert_id="E2E-TEST-ALERT-123").exists()
        self.assertFalse(alert_exists)

    def test_invalid_currency_error_handling(self):
        """Test error handling for invalid currency codes without mocking"""
        # Make request with a more obviously invalid currency, like a number
        response = self.client.get(f"{self.base_url}123/EUR/")

        # The server returns a 500 error when an invalid currency is provided
        self.assertEqual(response.status_code, 500)

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_unusual_currency_code(self, mock_requests_get):
        """Test with a 4-letter currency code which is rejected as invalid"""
        # The USDD currency code is rejected with a 400 error
        response = self.client.get(f"{self.base_url}USDD/EUR/")

        # The validation function correctly rejects the 4-letter code
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('detail', data)
        self.assertIn('Invalid currency code', data['detail'])


if __name__ == '__main__':
    unittest.main()
