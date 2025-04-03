import json
from django.urls import reverse
from rest_framework.test import APITestCase
from unittest.mock import patch
import requests
from datetime import datetime, timezone
from myapp.Models.financialNewsModel import FinancialNewsAlphaV
from rest_framework import status

class FetchFinancialNewsIntegrationTest(APITestCase):
    """Integration tests for the Financial News API endpoint"""

    def setUp(self):
        """Set up test environment"""
        self.url = reverse('fetch-financial-news')
        self.symbol = "AAPL"

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
    def test_fetch_financial_news_success(self, mock_get):
        """Test successful fetch and storage of financial news"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_news_data

        response = self.client.post(
            self.url,
            data=json.dumps({"symbol": self.symbol}),  
            content_type="application/json"
        )

        # Ensure the API successfully processes and returns a response
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_no_data(self, mock_get):
        """Test handling of empty news data response"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": []}

        response = self.client.post(
            self.url,
            data=json.dumps({"symbol": self.symbol}),  
            content_type="application/json"
        )

        # Expect 204 when no content is returned
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(response.content, b'')

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_api_error(self, mock_get):
        """Test handling of Alpha Vantage API errors"""
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        response = self.client.post(
            self.url,
            data=json.dumps({"symbol": self.symbol}),  
            content_type="application/json"
        )

        # Expect a 502 Bad Gateway response when an API error occurs
        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_duplicate_article(self, mock_get):
        """Test handling of duplicate news articles"""
        # Prepopulate the database with an existing article
        FinancialNewsAlphaV.objects.create(
            title="Test News Article",
            source="Test Source",
            url="http://test.com/article1",
            summary="This is a test summary",
            publication_date=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            sentiment_score=0.5,
            sentiment_label="positive"
        )

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_news_data

        response = self.client.post(
            self.url,
            data=json.dumps({"symbol": self.symbol}),  
            content_type="application/json"
        )

        # Ensure duplicate articles are not stored again
        self.assertEqual(response.status_code, status.HTTP_200_OK)
