from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch, MagicMock
from rest_framework import status


class CurrencyCollectorViewsTests(TestCase):
    """Tests for the Currency Collector API views."""

    def test_health_check(self):
        """Test health check endpoint returns expected message."""
        url = reverse('health-check')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json(), {"message": "Currency Collector API is running"})

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_get_latest_exchange_rate_success(self, mock_get):
        """Test successful retrieval of exchange rate data."""
        # Mock the Alpha Vantage API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Realtime Currency Exchange Rate": {
                "1. From_Currency Code": "EUR",
                "2. From_Currency Name": "Euro",
                "3. To_Currency Code": "USD",
                "4. To_Currency Name": "United States Dollar",
                "5. Exchange Rate": "1.08680000",
                "6. Last Refreshed": "2025-03-20 10:20:01",
                "7. Time Zone": "UTC",
                "8. Bid Price": "1.08670000",
                "9. Ask Price": "1.08690000"
            }
        }
        mock_get.return_value = mock_response

        url = reverse('currency-rate', kwargs={'base': 'EUR', 'target': 'USD'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Verify the response content matches our expected schema
        response_data = response.json()
        self.assertEqual(response_data['base'], "EUR")
        self.assertEqual(response_data['target'], "USD")
        self.assertEqual(response_data['rate'], "1.08680000")
        self.assertEqual(response_data['source'], "Alpha Vantage")
        self.assertIn('timestamp', response_data)

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_get_latest_exchange_rate_api_error(self, mock_get):
        """Test handling of Alpha Vantage API errors."""
        # Mock an error response from Alpha Vantage
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Error Message": "Invalid API call. Please retry or visit documentation for API usage."
        }
        mock_get.return_value = mock_response

        url = reverse('currency-rate', kwargs={'base': 'EUR', 'target': 'USD'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(response.json(), {"detail": "External API error: Unable to fetch exchange rate data."})

    def test_get_latest_exchange_rate_invalid_currency(self):
        """Test validation of currency codes."""
        url = reverse('currency-rate', kwargs={'base': 'EURUSD', 'target': 'USD'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            response.json(), {"detail": "Invalid currency code. Currency codes must be 3 characters."}
        )

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_get_latest_exchange_rate_connection_error(self, mock_get):
        """Test handling of connection errors to Alpha Vantage."""
        # Mock a connection error
        mock_get.side_effect = Exception("Connection error")

        url = reverse('currency-rate', kwargs={'base': 'EUR', 'target': 'USD'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(
            response.json(),
            {"detail": "An unexpected error occurred while processing your request."}
        )
