from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from myapp.models import CurrencyNewsAlphaV
from datetime import timedelta
from django.utils import timezone


class CurrencyNewsEndToEndTest(APITestCase):
    """End-to-end tests for the Currency News API endpoints"""

    def setUp(self):
        """Set up test environment"""
        self.fetch_url = reverse('fetch-currency-news', kwargs={'currency': 'USD'})
        self.list_url = reverse('currency-news-list')

        # Create some test data
        CurrencyNewsAlphaV.objects.create(
            title="USD Gains Strength",
            source="Financial Times",
            url="https://example.com/usd-strength",
            summary="US dollar shows strong performance",
            sentiment_score=0.75,
            sentiment_label="positive",
            publication_date=timezone.now(),
            currency="USD"
        )

    def test_fetch_currency_news_success(self):
        """Test successful fetching of currency news"""
        # Make the API request to fetch currency news
        response = self.client.post(self.fetch_url)

        # Assert the response status code
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Assert that the response contains the expected ADAGE fields
        self.assertIn("dataset_type", response.data)
        self.assertEqual(response.data["dataset_type"], "currency_news")
        self.assertIn("dataset_id", response.data)
        self.assertIn("events", response.data)

    def test_list_currency_news_success(self):
        """Test successful retrieval of currency news"""
        # Make the API request to list currency news
        response = self.client.get(self.list_url)

        # Assert the response status code
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Assert that the response contains the expected fields
        self.assertIn("dataset_type", response.data)
        self.assertEqual(response.data["dataset_type"], "currency_news")
        self.assertIn("dataset_id", response.data)
        self.assertIn("events", response.data)
        self.assertGreater(len(response.data["events"]), 0)

        # Verify the event data structure
        event = response.data["events"][0]
        self.assertIn("event_type", event)
        self.assertEqual(event["event_type"], "currency_news")
        self.assertIn("event_id", event)
        self.assertIn("attributes", event)

        # Verify the news attributes
        attributes = event["attributes"]
        self.assertIn("title", attributes)
        self.assertIn("source", attributes)
        self.assertIn("url", attributes)
        self.assertIn("summary", attributes)
        self.assertIn("sentiment_score", attributes)
        self.assertIn("sentiment_label", attributes)
        self.assertIn("currency", attributes)

        # Verify data types
        self.assertIsInstance(attributes["title"], str)
        self.assertIsInstance(float(attributes["sentiment_score"]), float)
        self.assertIsInstance(attributes["currency"], str)

    def test_filter_currency_news_by_currency(self):
        """Test filtering currency news by currency"""
        # Add news for another currency
        CurrencyNewsAlphaV.objects.create(
            title="EUR Weakens",
            source="Bloomberg",
            url="https://example.com/eur-weakness",
            summary="Euro shows weakness against USD",
            sentiment_score=-0.35,
            sentiment_label="negative",
            publication_date=timezone.now(),
            currency="EUR"
        )

        # Filter by USD
        response = self.client.get(f"{self.list_url}?currency=USD")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["events"]), 1)
        self.assertEqual(response.data["events"][0]["attributes"]["currency"], "USD")

    def test_filter_currency_news_by_sentiment(self):
        """Test filtering currency news by sentiment score"""
        # Filter by positive sentiment
        response = self.client.get(f"{self.list_url}?sentiment_score=0.7")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["events"]), 1)
        self.assertGreater(float(response.data["events"][0]["attributes"]["sentiment_score"]), 0.7)

    def test_pagination_of_currency_news(self):
        """Test limiting the number of returned news items"""
        # Add more test data
        for i in range(5):
            CurrencyNewsAlphaV.objects.create(
                title=f"USD News {i}",
                source=f"Source {i}",
                url=f"https://example.com/usd-news-{i}",
                summary=f"Summary {i}",
                sentiment_score=0.1 * i,
                sentiment_label="neutral",
                publication_date=timezone.now() - timedelta(days=i),
                currency="USD"
            )

        # Test limit parameter
        response = self.client.get(f"{self.list_url}?limit=3")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["events"]), 3)

    def test_empty_response_handling(self):
        """Test handling of empty responses"""
        # Clear test data
        CurrencyNewsAlphaV.objects.all().delete()

        # Test list endpoint with no data
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["events"]), 0)

        # Test filtering with no matches
        response = self.client.get(f"{self.list_url}?currency=JPY")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["events"]), 0)

    def test_response_time_performance(self):
        """Test response time for currency news endpoints"""
        # Create larger dataset for performance testing
        for i in range(50):
            CurrencyNewsAlphaV.objects.create(
                title=f"Performance Test News {i}",
                source="Test Source",
                url=f"https://example.com/perf-test-{i}",
                summary="Performance testing summary",
                sentiment_score=0.1 * (i % 10),
                sentiment_label="neutral",
                publication_date=timezone.now() - timedelta(days=i),
                currency=["USD", "EUR"][i % 2]
            )

        # Time the list endpoint
        import time
        start_time = time.time()
        response = self.client.get(f"{self.list_url}?limit=20")
        end_time = time.time()

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["events"]), 20)

        # Assert response time is reasonable (adjust threshold as needed)
        self.assertLess(end_time - start_time, 1.0)  # Should respond in under 1 second
