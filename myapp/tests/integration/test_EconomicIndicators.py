from django.urls import reverse
from rest_framework.test import APITestCase
from unittest.mock import patch
import requests


class AnnualIndicatorsIntegrationTest(APITestCase):
    """Integration tests for the Annual Economic Indicators API endpoints"""

    def setUp(self):
        """Set up test environment"""
        self.url = reverse('store-annual-economic-indicators')

    @patch('myapp.Service.economicIndicatorService.AnnualIndicatorsService.store_annual_indicators')
    def test_store_annual_indicators_success(self, mock_store):
        """Test successful storage and retrieval of annual indicators"""
        # Mock the service response
        mock_store.return_value = {
            "data_source": "Alpha Vantage",
            "dataset_type": "annual_economic_indicators",
            "dataset_id": "annual-indicators-2023-01-01",
            "time_object": {
                "timestamp": "2023-12-31T23:59:59.999Z",
                "timezone": "UTC"
            },
            "events": [
                {
                    "time_object": {
                        "timestamp": "2023-01-01",
                        "duration": 365,
                        "duration_unit": "days",
                        "timezone": "UTC"
                    },
                    "event_type": "economic_indicator",
                    "event_id": "AEI-20231231-abc123de",
                    "attributes": {
                        "real_gdp": 24989.50,
                        "inflation": 3.4,
                        "source": "Alpha Vantage"
                    }
                }
            ]
        }

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["dataset_type"], "annual_economic_indicators")
        self.assertEqual(response.data["dataset_id"], "annual-indicators-2023-01-01")
        self.assertEqual(len(response.data["events"]), 1)

        # Verify the event data
        event = response.data["events"][0]
        self.assertEqual(float(event["attributes"]["real_gdp"]), 24989.50)
        self.assertEqual(float(event["attributes"]["inflation"]), 3.4)

    @patch('myapp.Service.economicIndicatorService.AnnualIndicatorsService.store_annual_indicators')
    def test_store_annual_indicators_invalid_response(self, mock_store):
        """Test handling of invalid ADAGE format"""
        # Mock an invalid response
        mock_store.return_value = {
            "data_source": "Alpha Vantage",
            # Missing required fields
        }

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.data)

    @patch('myapp.Service.economicIndicatorService.AnnualIndicatorsService.store_annual_indicators')
    def test_store_annual_indicators_api_error(self, mock_store):
        """Test handling of external API errors"""
        # Mock a RequestException
        mock_store.side_effect = requests.exceptions.RequestException("External API down")

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.data)
        self.assertIn("Failed to fetch data from external API", response.data["error"])

    @patch('myapp.Service.economicIndicatorService.AnnualIndicatorsService.store_annual_indicators')
    def test_store_annual_indicators_general_error(self, mock_store):
        """Test handling of general errors"""
        # Mock a general exception
        mock_store.side_effect = Exception("Database error")

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.data)
        self.assertIn("Database error", response.data["error"])


class MonthlyIndicatorsIntegrationTest(APITestCase):
    """Integration tests for the Monthly Economic Indicators API endpoints"""

    def setUp(self):
        """Set up test environment"""
        self.url = reverse('store-monthly-economic-indicators')

    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.store_monthly_indicators')
    def test_store_monthly_indicators_success(self, mock_store):
        """Test successful storage and retrieval of monthly indicators"""
        # Mock the service response
        mock_store.return_value = {
            "data_source": "Alpha Vantage",
            "dataset_type": "monthly_economic_indicators",
            "dataset_id": "monthly-indicators-2024-03-01",
            "time_object": {
                "timestamp": "2024-03-31T23:59:59.999Z",
                "timezone": "UTC"
            },
            "events": [
                {
                    "time_object": {
                        "timestamp": "2024-03-01",
                        "duration": 1,
                        "duration_unit": "month",
                        "timezone": "UTC"
                    },
                    "event_type": "economic_indicator",
                    "event_id": "MEI-20240331-abc123de",
                    "attributes": {
                        "cpi": 307.8,
                        "unemployment_rate": 4.1,
                        "federal_funds_rate": 5.33,
                        "treasury_yield": 4.35,
                        "source": "Alpha Vantage"
                    }
                }
            ]
        }

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["dataset_type"], "monthly_economic_indicators")
        self.assertEqual(response.data["dataset_id"], "monthly-indicators-2024-03-01")
        self.assertEqual(len(response.data["events"]), 1)

        # Verify the event data
        event = response.data["events"][0]
        self.assertEqual(float(event["attributes"]["cpi"]), 307.8)
        self.assertEqual(float(event["attributes"]["unemployment_rate"]), 4.1)
        self.assertEqual(float(event["attributes"]["federal_funds_rate"]), 5.33)
        self.assertEqual(float(event["attributes"]["treasury_yield"]), 4.35)

    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.store_monthly_indicators')
    def test_store_monthly_indicators_invalid_response(self, mock_store):
        """Test handling of invalid ADAGE format"""
        # Mock an invalid response
        mock_store.return_value = {
            "data_source": "Alpha Vantage",
            # Missing required fields
        }

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.data)

    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.store_monthly_indicators')
    def test_store_monthly_indicators_api_error(self, mock_store):
        """Test handling of external API errors"""
        # Mock a RequestException
        mock_store.side_effect = requests.exceptions.RequestException("External API down")

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.data)
        self.assertIn("Failed to fetch data from external API", response.data["error"])

    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.store_monthly_indicators')
    def test_store_monthly_indicators_general_error(self, mock_store):
        """Test handling of general errors"""
        # Mock a general exception
        mock_store.side_effect = Exception("Database error")

        # Make the API request
        response = self.client.post(self.url)

        # Assert the response
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.data)
        self.assertIn("Database error", response.data["error"])
