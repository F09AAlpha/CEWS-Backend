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
        """Test successful retrieval of exchange rate data in ADAGE 3.0 format."""
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

        # Verify the response content matches ADAGE 3.0 data model
        response_data = response.json()

        # Check top-level structure
        self.assertIn('data_source', response_data)
        self.assertEqual(response_data['data_source'], "Alpha Vantage")
        self.assertIn('dataset_type', response_data)
        self.assertEqual(response_data['dataset_type'], "currency_exchange_rate")
        self.assertIn('dataset_id', response_data)
        self.assertIn('time_object', response_data)
        self.assertIn('events', response_data)

        # Check time object
        self.assertIn('timestamp', response_data['time_object'])
        self.assertIn('timezone', response_data['time_object'])
        self.assertEqual(response_data['time_object']['timezone'], "UTC")

        # Check that we have at least one event
        self.assertGreaterEqual(len(response_data['events']), 1)

        # Check event structure
        event = response_data['events'][0]
        self.assertIn('time_object', event)
        self.assertIn('event_type', event)
        self.assertEqual(event['event_type'], "currency_rate")
        self.assertIn('event_id', event)
        self.assertIn('attributes', event)

        # Check attributes
        attributes = event['attributes']
        self.assertEqual(attributes['base'], "EUR")
        self.assertEqual(attributes['target'], "USD")
        self.assertEqual(attributes['rate'], 1.0868)
        self.assertEqual(attributes['source'], "Alpha Vantage")
        self.assertEqual(attributes['bid_price'], "1.08670000")
        self.assertEqual(attributes['ask_price'], "1.08690000")

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

    @patch('myapp.Views.exchangeRateLatestViews.requests.get')
    def test_adage_data_model_integrity(self, mock_get):
        """Test that the ADAGE 3.0 Data Model structure is complete and consistent."""
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
                "7. Time Zone": "UTC"
            }
        }
        mock_get.return_value = mock_response

        url = reverse('currency-rate', kwargs={'base': 'EUR', 'target': 'USD'})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response_data = response.json()

        # Required fields at dataset level
        required_dataset_fields = ['data_source', 'dataset_type', 'dataset_id', 'time_object', 'events']
        for field in required_dataset_fields:
            self.assertIn(field, response_data, f"Required dataset field '{field}' is missing")

        # Required fields in time_object
        time_object = response_data['time_object']
        required_time_fields = ['timestamp', 'timezone']
        for field in required_time_fields:
            self.assertIn(field, time_object, f"Required time_object field '{field}' is missing")

        # Event structure
        events = response_data['events']
        self.assertGreaterEqual(len(events), 1, "At least one event should be present")

        event = events[0]
        required_event_fields = ['time_object', 'event_type', 'event_id', 'attributes']
        for field in required_event_fields:
            self.assertIn(field, event, f"Required event field '{field}' is missing")

        # Event time_object
        event_time = event['time_object']
        required_event_time_fields = ['timestamp', 'timezone', 'duration', 'duration_unit']
        for field in required_event_time_fields:
            self.assertIn(field, event_time, f"Required event time_object field '{field}' is missing")

        # Event attributes
        attributes = event['attributes']
        required_attribute_fields = ['base', 'target', 'rate', 'source']
        for field in required_attribute_fields:
            self.assertIn(field, attributes, f"Required attribute field '{field}' is missing")
