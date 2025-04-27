from django.db import connection
from django.urls import reverse
from rest_framework.test import APITestCase
from unittest.mock import patch
import requests


class HistoricalExchangeRatesV2IntegrationTest(APITestCase):
    """Integration tests for the V2 Historical Exchange Rates API endpoint"""

    def setUp(self):
        """Set up test environment"""
        self.from_currency = 'USD'
        self.to_currency = 'EUR'
        self.url = reverse('fetch-historical-exchange-rates-v2', args=[self.from_currency, self.to_currency])

        # Set up initial data in the database
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_exchange_rate_usd_eur (
                    id SERIAL,
                    date DATE NOT NULL PRIMARY KEY UNIQUE,
                    open DECIMAL(10, 5),
                    high DECIMAL(10, 5),
                    low DECIMAL(10, 5),
                    close DECIMAL(10, 5)
                )
            """)
            cursor.execute("""
                INSERT INTO historical_exchange_rate_usd_eur (date, open, high, low, close)
                VALUES ('2023-10-01', 1.1000, 1.2000, 1.0500, 1.1500)
                ON CONFLICT (date) DO NOTHING
            """)

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_no_new_data_not_stored(self, mock_get):
        """Test that no new data is stored if the latest data is already in the database"""
        # Mock response from Alpha Vantage with same latest date
        mock_response = {
            "Meta Data": {
                "2. From Symbol": self.from_currency,
                "3. To Symbol": self.to_currency
            },
            "Time Series FX (Daily)": {
                "2023-10-01": {
                    "1. open": "1.1000",
                    "2. high": "1.2000",
                    "3. low": "1.0500",
                    "4. close": "1.1500"
                }
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Make API call
        response = self.client.post(self.url)

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("data_source", response.data)
        self.assertIn("event", response.data)

        # Check database still has only one record
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM historical_exchange_rate_usd_eur")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_fetch_exchange_rates_success(self, mock_get):
        """Test successful retrieval and insertion of new historical exchange rates"""
        # Mock response with a NEW date
        mock_response = {
            "Meta Data": {
                "2. From Symbol": self.from_currency,
                "3. To Symbol": self.to_currency
            },
            "Time Series FX (Daily)": {
                "2023-10-02": {
                    "1. open": "1.2000",
                    "2. high": "1.3000",
                    "3. low": "1.0600",
                    "4. close": "1.1600"
                }
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data_source"], "Alpha Vantage")
        self.assertEqual(response.data["dataset_type"], "historical_currency_exchange_rate")
        self.assertIn("event", response.data)
        self.assertTrue(response.data["event"][0]["attributes"]["data"])

        # Check new data was inserted
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM historical_exchange_rate_usd_eur")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_fetch_exchange_rates_no_data(self, mock_get):
        """Test handling of empty data from API"""
        # Mock empty API response
        mock_response = {
            "Meta Data": {},
            "Time Series FX (Daily)": {}
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.data)

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_fetch_exchange_rates_api_error(self, mock_get):
        """Test handling of external API failure"""
        mock_get.side_effect = requests.exceptions.RequestException("Service Unavailable")

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.data)
        self.assertIn("Failed to fetch data from external API", response.data["error"])
