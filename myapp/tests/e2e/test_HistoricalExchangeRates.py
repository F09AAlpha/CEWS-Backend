from django.db import connection
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status

class HistoricalExchangeRatesEndToEndTest(APITestCase):
    """End-to-end tests for the Historical Exchange Rates API endpoint"""

    def setUp(self):
        """Set up test environment"""
        self.from_currency = 'USD'
        self.to_currency = 'EUR'
        self.url = reverse('fetch-historical-exchange-rates', args=[self.from_currency, self.to_currency])

    def test_fetch_exchange_rates_success(self):
        """Test successful retrieval and storage of historical exchange rates"""
        # Make the API request to fetch historical exchange rates
        response = self.client.post(self.url)

        # Assert the response status code
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Assert that the response contains the expected fields
        self.assertIn("data_source", response.data)
        self.assertIn("dataset_type", response.data)
        self.assertIn("dataset_id", response.data)
        self.assertIn("event", response.data)
        self.assertGreater(len(response.data["event"]), 0)

        # Verify the event data
        event = response.data["event"][0]
        self.assertIn("attributes", event)
        self.assertIn("base", event["attributes"])
        self.assertIn("target", event["attributes"])
        self.assertIn("data", event["attributes"])

        # Verify the event attributes (e.g., base and target currencies)
        self.assertEqual(event["attributes"]["base"], self.from_currency)
        self.assertEqual(event["attributes"]["target"], self.to_currency)

    def test_fetch_exchange_rates_no_new_data(self):
        """Test retrieval when no new data is available"""
        # First request to populate the database
        self.client.post(self.url)

        # Get length of new inserted data
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM historical_exchange_rate_usd_eur")
            count1 = cursor.fetchone()[0]

        # Second request should find no new data
        response = self.client.post(self.url)

        # Assert the response status code
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Assert that the response contains the expected fields
        self.assertIn("data_source", response.data)
        self.assertIn("dataset_type", response.data)
        self.assertIn("dataset_id", response.data)
        self.assertIn("event", response.data)

        # Verify that no new data was added
        #self.assertEqual(len(response.data["event"]), 0)
        
        # Verify that no new data was inserted
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM historical_exchange_rate_usd_eur")
            count2 = cursor.fetchone()[0]
        
        self.assertEqual(count1,count2)