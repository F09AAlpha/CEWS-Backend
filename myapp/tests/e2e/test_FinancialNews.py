import json
from django.test import TestCase
from unittest.mock import patch
from django.urls import reverse
from rest_framework import status
from myapp.Models.financialNewsModel import FinancialNewsAlphaV


class FetchFinancialNewsE2ETest(TestCase):

    def setUp(self):
        """Set up test environment with API endpoint and necessary parameters."""
        self.url = reverse('fetch-financial-news')
        self.test_symbol = "AAPL"

        self.mock_news_data = {
            "feed": [
                {
                    "title": "Test News Article",
                    "source": "Test Source",
                    "url": "http://test.com/article1",
                    "summary": "This is a test summary",
                    "time_published": "20240101T120000",
                    "overall_sentiment_score": 0.5,
                    "overall_sentiment_label": "positive",
                }
            ]
        }

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_success_e2e(self, mock_get):
        """Test fetching and storing financial news successfully."""

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_news_data

        response = self.client.post(
            self.url,
            json.dumps({"symbol": self.test_symbol}),
            content_type="application/json"
        )

        # Allow either 201 (new data stored) or 200 (existing data returned)
        self.assertIn(response.status_code, [status.HTTP_201_CREATED, status.HTTP_200_OK])

        # Ensure news articles are stored in the database
        stored_articles = FinancialNewsAlphaV.objects.all()
        self.assertGreater(len(stored_articles), 0)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_no_data_e2e(self, mock_get):
        """Test behavior when no news articles are available."""

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": []}

        response = self.client.post(
            self.url,
            json.dumps({"symbol": "RANDOM123"}),
            content_type="application/json"
        )

        # Expecting 204 when no content is available
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_duplicate_e2e(self, mock_get):
        """Test that duplicate news articles are not stored multiple times."""

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_news_data

        # First request to store the news article
        initial_response = self.client.post(
            self.url,
            json.dumps({"symbol": self.test_symbol}),
            content_type="application/json"
        )
        self.assertIn(initial_response.status_code, [status.HTTP_201_CREATED, status.HTTP_200_OK])

        # Second request should not store a duplicate
        second_response = self.client.post(
            self.url,
            json.dumps({"symbol": self.test_symbol}),
            content_type="application/json"
        )
        self.assertEqual(second_response.status_code, status.HTTP_200_OK)

        # Ensure only unique articles are stored
        stored_articles = FinancialNewsAlphaV.objects.all()
        self.assertGreater(len(stored_articles), 0)

        # Validate uniqueness of stored articles based on URL
        unique_urls = set(article.url for article in stored_articles)
        self.assertEqual(len(unique_urls), len(stored_articles))
