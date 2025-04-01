from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status


class AnnualIndicatorsEndToEndTest(APITestCase):
    """End-to-end tests for the Annual Economic Indicators API endpoint"""

    def setUp(self):
        """Set up test environment"""
        self.url = reverse('store-annual-economic-indicators')  # URL for the annual indicators endpoint

    def test_store_annual_indicators_success(self):
        """Test successful storage and retrieval of annual economic indicators"""
        # Make the API request to store annual indicators
        response = self.client.post(self.url)

        # Assert the response status code
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Assert that the response contains the expected fields
        self.assertIn("dataset_type", response.data)
        self.assertIn("dataset_id", response.data)
        self.assertIn("events", response.data)
        self.assertGreater(len(response.data["events"]), 0)

        # Verify the event data
        event = response.data["events"][0]
        self.assertIn("attributes", event)
        self.assertIn("real_gdp", event["attributes"])
        self.assertIn("inflation", event["attributes"])

        # Verify the event attributes (e.g., real GDP and inflation)
        self.assertIsInstance(float(event["attributes"]["real_gdp"]), float)
        self.assertIsInstance(float(event["attributes"]["inflation"]), float)
