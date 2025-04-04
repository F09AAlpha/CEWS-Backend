from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from django.utils import timezone
from datetime import datetime
from unittest.mock import patch, MagicMock
from myapp.models import CurrencyNewsAlphaV
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CEWS.settings')


class CurrencyNewsIntegrationTest(APITestCase):
    """Integration tests for Currency News API endpoints"""

    def setUp(self):
        self.client = APIClient()
        self.list_url = reverse('currency-news-list')
        self.fetch_url = reverse('fetch-currency-news', kwargs={'currency': 'USD'})

        # Create test data
        self.test_news = CurrencyNewsAlphaV.objects.create(
            title="USD Strengthens Against Euro",
            source="Financial Times",
            url="https://example.com/usd-strength",
            summary="The US dollar showed strong gains against the Euro today",
            sentiment_score=0.75,
            sentiment_label="positive",
            publication_date=timezone.now(),
            currency="USD"
        )

    @patch('myapp.Views.currencyNewsView.requests.get')
    def test_fetch_currency_news_success(self, mock_get):
        """Test successful fetching of currency news from external API"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "feed": [
                {
                    "title": "USD Gains Ground",
                    "source": "Bloomberg",
                    "url": "https://example.com/usd-gains",
                    "summary": "USD rises on positive economic data",
                    "time_published": "20240101T120000",
                    "overall_sentiment_score": 0.65,
                    "overall_sentiment_label": "positive",
                    "ticker_sentiment": [
                        {"ticker": "USD", "relevance_score": "0.95"}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response

        response = self.client.post(self.fetch_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['dataset_type'], "currency_news")
        self.assertTrue(CurrencyNewsAlphaV.objects.filter(url="https://example.com/usd-gains").exists())

    def test_list_currency_news_success(self):
        """Test successful retrieval of currency news"""
        response = self.client.get(f"{self.list_url}?currency=USD")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['dataset_type'], "currency_news")
        self.assertEqual(len(response.data['events']), 1)
        self.assertEqual(response.data['events'][0]['attributes']['title'], 
                         "USD Strengthens Against Euro")

    def test_list_currency_news_filtering(self):
        """Test filtering by currency and sentiment"""
        # Create additional test data
        CurrencyNewsAlphaV.objects.create(
            title="EUR Weakens",
            source="Reuters",
            url="https://example.com/eur-weakness",
            summary="Euro falls against major currencies",
            sentiment_score=-0.45,
            sentiment_label="negative",
            publication_date=timezone.now(),
            currency="EUR"
        )

        # Test currency filter
        response = self.client.get(f"{self.list_url}?currency=USD")
        self.assertEqual(len(response.data['events']), 1)

        # Test sentiment filter
        response = self.client.get(f"{self.list_url}?sentiment_score=0.7")
        self.assertEqual(len(response.data['events']), 1)

        # Test combined filter
        response = self.client.get(f"{self.list_url}?currency=USD&sentiment_score=0.7")
        self.assertEqual(len(response.data['events']), 1)

    def test_list_currency_news_empty(self):
        """Test empty response handling"""
        response = self.client.get(f"{self.list_url}?currency=JPY")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['events']), 0)

    @patch('myapp.Serializers.currencyNewsSerializer.AdageNewsDatasetSerializer.is_valid')
    def test_list_currency_news_fallback(self, mock_is_valid):
        """Test fallback to regular serializer"""
        mock_is_valid.return_value = False

        response = self.client.get(self.list_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
        self.assertEqual(len(response.data), 1)

    @patch('myapp.Views.currencyNewsView.uuid.uuid4')
    @patch('myapp.Views.currencyNewsView.datetime')
    def test_deterministic_response_format(self, mock_datetime, mock_uuid):
        """Test deterministic response format with mocked values"""
        # Setup mocks
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_time
        mock_uuid.return_value = MagicMock(hex='testuuid1234')

        response = self.client.get(f"{self.list_url}?currency=USD")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['dataset_id'], "currency-news-USD-20240101")
        self.assertTrue(
            response.data['events'][0]['event_id'].startswith("CN-20240101-testuuid")
        )
