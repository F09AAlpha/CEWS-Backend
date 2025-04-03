from django.db import connection
from django.urls import reverse
from rest_framework.test import APITestCase
from unittest.mock import patch
import requests


class HistoricalExchangeRatesIntegrationTest(APITestCase):
    """Integration tests for the Historical Exchange Rates API endpoint"""

    def setUp(self):
        """Set up test environment"""
        self.from_currency = 'USD'
        self.to_currency = 'EUR'
        self.url = reverse('fetch-historical-exchange-rates', args=[self.from_currency, self.to_currency])
        
        # Set up initial data in the database
        with connection.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS historical_exchange_rate_usd_eur (
                    id SERIAL,
                    date DATE NOT NULL PRIMARY KEY UNIQUE,
                    open DECIMAL(10, 5),
                    high DECIMAL(10, 5),
                    low DECIMAL(10, 5),
                    close DECIMAL(10, 5)
                )
            """)
            cursor.execute(f"""
                INSERT INTO historical_exchange_rate_usd_eur (date, open, high, low, close)
                VALUES ('2023-10-01', 1.1000, 1.2000, 1.0500, 1.1500)
                ON CONFLICT (date) DO NOTHING
            """)

    @patch('requests.get')
    def test_no_new_data_not_stored(self, mock_get):
        """Test that no new data is stored if the latest data is already in the database"""
        # Mock the response from Alpha Vantage with the same latest date
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

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 200)
        self.assertIn("data_source", response.data)
        self.assertIn("event", response.data)

        # Verify that no new data was inserted
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM historical_exchange_rate_usd_eur")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)  # Ensure only the initial data is present

    @patch('requests.get')
    def test_fetch_exchange_rates_success(self, mock_get):
        """Test successful retrieval and storage of historical exchange rates"""
        # Mock the response from Alpha Vantage
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

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data_source"], "Alpha Vantage")
        self.assertEqual(response.data["dataset_type"], "historical_currency_exchange_rate")
        self.assertIn("event", response.data)

    @patch('requests.get')
    def test_fetch_exchange_rates_no_data(self, mock_get):
        """Test handling of no data available"""
        # Mock an empty response from Alpha Vantage
        mock_response = {
            "Meta Data": {},
            "Time Series FX (Daily)": {}
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.data)

    @patch('requests.get')
    def test_fetch_exchange_rates_api_error(self, mock_get):
        """Test handling of external API errors"""
        # Mock a RequestException
        mock_get.side_effect = requests.exceptions.RequestException("External API down")

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.data)
        self.assertIn("Failed to fetch data from external API", response.data["error"])
