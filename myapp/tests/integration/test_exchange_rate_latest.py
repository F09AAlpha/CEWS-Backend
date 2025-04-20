from django.test import TestCase
from rest_framework.test import APIClient
from unittest.mock import patch, MagicMock
from decimal import Decimal

from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert


class ExchangeRateViewIntegrationTests(TestCase):
    """Integration tests for the CurrencyRateView"""

    def setUp(self):
        self.client = APIClient()
        self.base_url = '/api/v1/currency/rates/'

        # Create test alert
        self.test_alert = ExchangeRateAlert.objects.create(
            alert_id="TEST-ALERT-123",
            base="USD",
            target="EUR",
            alert_type="above",
            threshold=Decimal('0.90'),
            email="test@example.com"
        )

    def tearDown(self):
        # Clean up created objects
        ExchangeRateAlert.objects.filter(alert_id="TEST-ALERT-123").delete()

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_exchange_rate_view_success(self, mock_requests_get):
        """Test successful API response with mocked external API call"""
        # Mock the external API response with rate BELOW threshold (0.90)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Realtime Currency Exchange Rate": {
                "1. From_Currency Code": "USD",
                "3. To_Currency Code": "EUR",
                "5. Exchange Rate": "0.85",  # Below our threshold of 0.90
                "6. Last Refreshed": "2023-04-01 12:00:00",
                "8. Bid Price": "0.84",
                "9. Ask Price": "0.86"
            }
        }
        mock_requests_get.return_value = mock_response

        # Make request to our API using direct URL
        response = self.client.get(f"{self.base_url}USD/EUR/")

        # Assertions
        self.assertEqual(response.status_code, 200)

        # Verify structure of response
        data = response.json()
        self.assertEqual(data['data_source'], "Alpha Vantage")
        self.assertEqual(data['dataset_type'], "currency_exchange_rate")
        self.assertEqual(len(data['events']), 1)

        # Verify event data
        event = data['events'][0]
        self.assertEqual(event['event_type'], "currency_rate")
        self.assertEqual(event['attributes']['base'], "USD")
        self.assertEqual(event['attributes']['target'], "EUR")
        self.assertEqual(event['attributes']['rate'], 0.85)

        # Verify external API was called correctly
        mock_requests_get.assert_called_once()
        args, kwargs = mock_requests_get.call_args
        self.assertEqual(kwargs['params']['from_currency'], 'USD')
        self.assertEqual(kwargs['params']['to_currency'], 'EUR')

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_alert_triggering(self, mock_requests_get):
        """Test that alerts are triggered when threshold is reached"""
        # Mock the external API response with rate above threshold (0.90)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Realtime Currency Exchange Rate": {
                "1. From_Currency Code": "USD",
                "3. To_Currency Code": "EUR",
                "5. Exchange Rate": "0.92",
                "6. Last Refreshed": "2023-04-01 12:00:00",
                "8. Bid Price": "0.91",
                "9. Ask Price": "0.93"
            }
        }
        mock_requests_get.return_value = mock_response

        # Make request to our API using direct URL
        response = self.client.get(f"{self.base_url}USD/EUR/")

        # Verify response is successful
        self.assertEqual(response.status_code, 200)

        # Verify alert was deleted after triggering
        self.assertEqual(ExchangeRateAlert.objects.filter(alert_id="TEST-ALERT-123").count(), 0)

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_invalid_currency_code(self, mock_requests_get):
        """Test error handling for invalid currency codes"""
        # Make request with invalid currency code (too long)
        response = self.client.get(f"{self.base_url}USDD/EUR/")

        # Assertions
        self.assertEqual(response.status_code, 400)
        self.assertIn('detail', response.json())
        self.assertIn('Invalid currency code', response.json()['detail'])

        # External API should not be called
        mock_requests_get.assert_not_called()

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_external_api_error(self, mock_requests_get):
        """Test error handling when external API fails"""
        # Mock the external API error
        mock_requests_get.side_effect = Exception("Connection error")

        # Make request to our API using direct URL
        response = self.client.get(f"{self.base_url}USD/EUR/")

        # Assertions
        self.assertEqual(response.status_code, 500)
        self.assertIn('detail', response.json())
