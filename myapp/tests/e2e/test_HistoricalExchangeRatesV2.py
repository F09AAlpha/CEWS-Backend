from django.db import connection
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status


class HistoricalExchangeRatesV2EndToEndTest(APITestCase):
    """End-to-end tests for the Historical Exchange Rates V2 API endpoint"""

    def setUp(self):
        """Set up test environment"""
        self.from_currency = 'USD'
        self.to_currency = 'EUR'
        self.url = reverse('fetch-historical-exchange-rates-v2', args=[self.from_currency, self.to_currency])
        self.table_name = f"historical_exchange_rate_{self.from_currency.lower()}_{self.to_currency.lower()}"

    def test_fetch_exchange_rates_v2_success(self):
        """Test successful retrieval and storage of historical exchange rates using V2"""
        # Make the API request to fetch historical exchange rates
        response = self.client.post(self.url)

        # Assert the response status code
        self.assertIn(response.status_code, [status.HTTP_201_CREATED, status.HTTP_200_OK])

        # Assert that the response contains the expected fields
        self.assertIn("data_source", response.data)
        self.assertIn("dataset_type", response.data)
        self.assertIn("dataset_id", response.data)
        self.assertIn("time_object", response.data)
        self.assertIn("event", response.data)
        self.assertGreater(len(response.data["event"]), 0)

        # Verify the event structure
        event = response.data["event"][0]
        self.assertIn("event_type", event)
        self.assertIn("event_id", event)
        self.assertIn("attributes", event)

        attributes = event["attributes"]
        self.assertIn("base", attributes)
        self.assertIn("target", attributes)
        self.assertIn("data", attributes)

        # Check currencies match
        self.assertEqual(attributes["base"], self.from_currency)
        self.assertEqual(attributes["target"], self.to_currency)

        # Check that the data list is not empty
        self.assertIsInstance(attributes["data"], list)
        self.assertGreater(len(attributes["data"]), 0)

    def test_fetch_exchange_rates_v2_no_new_data(self):
        """Test retrieval when no new data is available using V2"""
        # First request to populate the database
        self.client.post(self.url)

        # Get count of inserted data
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count_before = cursor.fetchone()[0]

        # Second request to fetch again
        response = self.client.post(self.url)

        # Should return 200 OK when no new data
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Assert standard response fields
        self.assertIn("data_source", response.data)
        self.assertIn("dataset_type", response.data)
        self.assertIn("dataset_id", response.data)
        self.assertIn("time_object", response.data)
        self.assertIn("event", response.data)

        # Ensure no extra records inserted
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count_after = cursor.fetchone()[0]

        self.assertEqual(count_before, count_after)

    def test_fetch_exchange_rates_v2_invalid_currency(self):
        """Test retrieval with invalid currencies"""
        invalid_url = reverse('fetch-historical-exchange-rates-v2', args=['INVALID', 'XXX'])
        
        response = self.client.post(invalid_url)

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertIn("error", response.data)

