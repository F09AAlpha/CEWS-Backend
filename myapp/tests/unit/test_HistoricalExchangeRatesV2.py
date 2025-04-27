from django.test import TestCase
from unittest.mock import patch
from rest_framework.test import APIRequestFactory
from rest_framework import status
from django.urls import reverse
import requests

from myapp.Views.historicalExchangeRatesViewV2 import FetchHistoricalCurrencyExchangeRatesV2  # adjust import


class FetchHistoricalCurrencyExchangeRatesV2Test(TestCase):
    """Unit tests for FetchHistoricalCurrencyExchangeRatesV2"""

    def setUp(self):
        """Setup test environment"""
        self.factory = APIRequestFactory()
        self.view = FetchHistoricalCurrencyExchangeRatesV2.as_view()
        self.from_currency = "USD"
        self.to_currency = "EUR"
        self.url = reverse('fetch-historical-exchange-rates-v2', kwargs={
            'from_currency': self.from_currency,
            'to_currency': self.to_currency
        })

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_fetch_historical_rates_success(self, mock_get):
        """Test successful fetch and insertion of historical rates"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "Meta Data": {
                "2. From Symbol": "USD",
                "3. To Symbol": "EUR"
            },
            "Time Series FX (Daily)": {
                "2024-04-25": {
                    "1. open": "1.1000",
                    "2. high": "1.1100",
                    "3. low": "1.0900",
                    "4. close": "1.1050"
                }
            }
        }

        request = self.factory.post(self.url)
        response = self.view(request, from_currency=self.from_currency, to_currency=self.to_currency)

        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_201_CREATED])
        self.assertEqual(response.data['data_source'], 'Alpha Vantage')
        self.assertEqual(response.data['event'][0]['attributes']['base'], 'USD')
        self.assertEqual(response.data['event'][0]['attributes']['target'], 'EUR')

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_fetch_historical_rates_invalid_currency(self, mock_get):
        """Test handling of invalid currency request"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "Meta Data": {
                "2. From Symbol": "USD",
                "3. To Symbol": "EUR"
            },
            "Time Series FX (Daily)": {}  # empty time series
        }

        request = self.factory.post(self.url)
        response = self.view(request, from_currency=self.from_currency, to_currency=self.to_currency)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_fetch_historical_rates_api_failure(self, mock_get):
        """Test API failure"""
        mock_get.side_effect = requests.exceptions.RequestException("API Failure")

        request = self.factory.post(self.url)
        response = self.view(request, from_currency=self.from_currency, to_currency=self.to_currency)

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertIn('error', response.data)

    @patch('myapp.Views.historicalExchangeRatesViewV2.requests.get')
    def test_fetch_historical_rates_partial_data(self, mock_get):
        """Test handling when some fields are missing in the API response"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "Meta Data": {
                "2. From Symbol": "USD",
                "3. To Symbol": "JPY"
            },
            "Time Series FX (Daily)": {
                "2024-04-24": {
                    "1. open": "110.00",
                    # missing "2. high", "3. low", "4. close"
                }
            }
        }

        request = self.factory.post(self.url)
        response = self.view(request, from_currency="USD", to_currency="JPY")

        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_201_CREATED])
        event_data = response.data['event'][0]['attributes']['data']
        self.assertEqual(len(event_data), 1)
        self.assertEqual(event_data[0]['high'], 0.0)  # default to 0.0 for missing fields
