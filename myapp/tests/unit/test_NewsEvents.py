from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from django.utils import timezone
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from myapp.models import CurrencyNewsAlphaV
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CEWS.settings')


class CurrencyNewsListViewTest(APITestCase):
    """Test suite for CurrencyNewsListView which handles currency news events"""

    @classmethod
    def setUpTestData(cls):
        cls.url = reverse('currency-news-list')  # Just the base URL without parameters
        cls.client = APIClient()

        # Create currency news items with different attributes
        CurrencyNewsAlphaV.objects.create(
            title="USD rises against major currencies",
            source="Financial Times",
            url="https://example.com/usd-rises",
            summary="The US dollar gained strength against major currencies today.",
            publication_date=timezone.now() - timedelta(days=1),
            sentiment_score=0.75,
            sentiment_label="positive",
            currency="USD"
        )

        CurrencyNewsAlphaV.objects.create(
            title="EUR weakens following ECB announcement",
            source="Bloomberg",
            url="https://example.com/eur-weakens",
            summary="The Euro weakened after the ECB announced new policies.",
            publication_date=timezone.now() - timedelta(days=2),
            sentiment_score=0.25,
            sentiment_label="neutral",
            currency="EUR"
        )

        CurrencyNewsAlphaV.objects.create(
            title="GBP drops to three-month low",
            source="Reuters",
            url="https://example.com/gbp-drops",
            summary="The British pound fell to a three-month low against the USD.",
            publication_date=timezone.now() - timedelta(days=3),
            sentiment_score=-0.6,
            sentiment_label="negative",
            currency="GBP"
        )

        # Add more with specific sentiment scores for filtering tests
        CurrencyNewsAlphaV.objects.create(
            title="JPY sentiment test exactly 0.5",
            source="Test Source",
            url="https://example.com/jpy-test",
            summary="Test summary for JPY with sentiment score 0.5",
            publication_date=timezone.now() - timedelta(days=4),
            sentiment_score=0.5,
            sentiment_label="positive",
            currency="JPY"
        )

        CurrencyNewsAlphaV.objects.create(
            title="JPY sentiment test 0.52 - close to 0.5",
            source="Test Source",
            url="https://example.com/jpy-test-2",
            summary="Test summary for JPY with sentiment score 0.52",
            publication_date=timezone.now() - timedelta(days=4),
            sentiment_score=0.52,
            sentiment_label="positive",
            currency="JPY"
        )

    def test_get_all_news_no_params(self):
        """Test retrieving all news events with no parameters."""
        response = self.client.get(self.url)

        # Check response status
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Ensure we got ADAGE format
        self.assertIsInstance(response.data, dict)

        # Check default limit of 10 is applied
        self.assertLessEqual(len(response.data['events']), 10)

        # Verify response structure matches ADAGE 3.0 format
        self.assertIn("data_source", response.data)
        self.assertIn("dataset_type", response.data)
        self.assertIn("dataset_id", response.data)
        self.assertIn("time_object", response.data)
        self.assertIn("events", response.data)

        # Check at least our created test events are present
        self.assertGreaterEqual(len(response.data["events"]), 5)

    def test_filter_by_currency(self):
        """Test filtering news by currency parameter."""
        response = self.client.get(f"{self.url}?currency=USD")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, dict)

        # Verify all returned events have the requested currency
        for event in response.data['events']:
            self.assertEqual(event["attributes"]["currency"], "USD")

        # Check we got the expected number of USD events
        self.assertEqual(len(response.data['events']), 1)

    def test_filter_by_sentiment(self):
        """Test filtering news by sentiment_score parameter."""
        # Test exact sentiment with range filter
        response = self.client.get(f"{self.url}?sentiment_score=0.5")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, dict)

        # Should get events with sentiment between 0.45 and 0.55
        self.assertGreaterEqual(len(response.data["events"]), 1)

        # Verify sentiment scores are within the expected range
        for event in response.data["events"]:
            sentiment = event["attributes"]["sentiment_score"]
            self.assertGreaterEqual(sentiment, 0.45)
            self.assertLessEqual(sentiment, 0.55)

    def test_invalid_sentiment_parameter(self):
        """Test handling of invalid sentiment parameter."""
        # Test with non-numeric sentiment value
        response = self.client.get(f"{self.url}?sentiment_score=invalid")

        # Should return 400 status for invalid sentiment
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("detail", response.data)

    def test_custom_limit(self):
        """Test limiting the number of returned results."""
        response = self.client.get(f"{self.url}?limit=2")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, dict)
        self.assertEqual(len(response.data['events']), 2)

    def test_multiple_filters(self):
        """Test applying multiple filters simultaneously."""
        # Create an additional JPY entry with positive sentiment
        CurrencyNewsAlphaV.objects.create(
            title="Second JPY news with 0.5 sentiment",
            source="Test Source",
            url="https://example.com/jpy-test-3",
            summary="Another test summary for JPY with sentiment score 0.5",
            publication_date=timezone.now() - timedelta(days=5),
            sentiment_score=0.5,
            sentiment_label="positive",
            currency="JPY"
        )

        # Test filtering by both currency and sentiment
        response = self.client.get(f"{self.url}?currency=JPY&sentiment_score=0.5")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, dict)

        # Check all returned events match both filters
        for event in response.data['events']:
            self.assertEqual(event["attributes"]["currency"], "JPY")
            sentiment = event["attributes"]["sentiment_score"]
            self.assertGreaterEqual(sentiment, 0.45)
            self.assertLessEqual(sentiment, 0.55)

    def test_empty_response(self):
        """Test handling of empty queryset."""
        # Request a currency that doesn't exist in our test data
        response = self.client.get(f"{self.url}?currency=XYZ")

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # For empty response, it could either be an empty list or
        # an ADAGE format with empty events array
        if isinstance(response.data, dict):
            self.assertEqual(len(response.data['events']), 0)
        else:
            self.assertEqual(len(response.data), 0)

    def test_adage_format_structure(self):
        """Test that the response follows the ADAGE 3.0 format structure."""
        response = self.client.get(self.url)

        # Verify the ADAGE structure is consistent
        self.assertIsInstance(response.data, dict)
        self.assertEqual(response.data['dataset_type'], "currency_news")
        self.assertIn("currency-news", response.data["dataset_id"])

        # Check time object format
        self.assertIn("timestamp", response.data["time_object"])
        self.assertEqual(response.data["time_object"]["timezone"], "UTC")

        # Check event structure
        event = response.data["events"][0]
        self.assertIn("event_type", event)
        self.assertEqual(event["event_type"], "currency_news")
        self.assertIn("event_id", event)

        # Check attributes
        self.assertIn("attributes", event)
        self.assertIn("title", event["attributes"])
        self.assertIn("source", event["attributes"])
        self.assertIn("url", event["attributes"])
        self.assertIn("summary", event["attributes"])
        self.assertIn("sentiment_score", event["attributes"])
        self.assertIn("sentiment_label", event["attributes"])
        self.assertIn("currency", event["attributes"])

    @patch('myapp.Serializers.currencyNewsSerializer.AdageNewsDatasetSerializer.is_valid')
    def test_fallback_to_regular_serializer(self, mock_is_valid):
        """Test fallback to regular serializer when ADAGE validation fails."""
        # Mock the is_valid method to return False
        mock_is_valid.return_value = False

        response = self.client.get(self.url)

        # Should still return 200 but use the regular serializer
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Response should be a list of serialized news items, not ADAGE format
        self.assertTrue(isinstance(response.data, list))

    @patch('myapp.Views.currencyNewsView.uuid.uuid4')
    @patch('myapp.Views.currencyNewsView.datetime')
    def test_deterministic_response(self, mock_datetime, mock_uuid):
        """Test that response structure is deterministic with fixed time and UUID."""
        # Mock datetime.now() to return a fixed datetime
        fixed_datetime = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_datetime
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        # Mock UUID to return deterministic values
        mock_uuid.return_value = MagicMock(hex='abcdef1234567890')

        # Get response with deterministic values
        response = self.client.get(f"{self.url}?currency=USD")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, dict)

        # Verify deterministic dataset_id
        expected_date_str = "20250101"
        self.assertEqual(response.data["dataset_id"], f"currency-news-USD-{expected_date_str}")

        # Verify deterministic event_id pattern
        for event in response.data["events"]:
            self.assertTrue(event["event_id"].startswith(f"CN-{expected_date_str}-abcdef12"))
