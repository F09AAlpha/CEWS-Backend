from django.test import TestCase
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from myapp.Models.financialNewsModel import FinancialNewsAlphaV
from myapp.Views.financialNewsView import FetchFinancialNewsView
from rest_framework.test import APIRequestFactory
from rest_framework import status
from django.urls import reverse
import requests
import json

class FetchFinancialNewsViewTest(TestCase):
    """Unit tests for FetchFinancialNewsView"""

    def setUp(self):
        """Setup test environment"""
        self.factory = APIRequestFactory()
        self.view = FetchFinancialNewsView.as_view()
        self.url = reverse('fetch-financial-news')
        self.test_symbol = "AAPL"

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_success(self, mock_get):
        """Test successful fetch and storage of financial news"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "feed": [{
                "title": "AAPL reports record earnings",
                "source": "Financial Times",
                "url": "https://example.com/news/1",
                "summary": "Apple Inc. reported record quarterly earnings.",
                "overall_sentiment_score": "0.85",
                "overall_sentiment_label": "positive",
                "time_published": "20240301T120000"
            }]
        }

        request = self.factory.post(self.url, {"symbol": self.test_symbol}, format='json')
        response = self.view(request)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(FinancialNewsAlphaV.objects.count(), 1)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_custom_symbol(self, mock_get):
        """Test fetching news for a different stock symbol"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "feed": [{
                "title": "MSFT cloud business growth",
                "source": "CNBC",
                "url": "https://example.com/news/3",
                "summary": "Microsoft's cloud business reports growth.",
                "overall_sentiment_score": "0.72",
                "overall_sentiment_label": "positive",
                "time_published": "20240301T110000"
            }]
        }

        request = self.factory.post(self.url, {"symbol": "MSFT"}, format='json')
        response = self.view(request)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(FinancialNewsAlphaV.objects.first().symbol, 'MSFT')

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_api_failure(self, mock_get):
        """Test API failure handling"""
        mock_get.return_value.status_code = 500
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")

        request = self.factory.post(self.url, {"symbol": self.test_symbol}, format='json')
        response = self.view(request)

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_no_data(self, mock_get):
        """Test handling of empty news response"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": []}

        response = self.client.post(self.url, json.dumps({"symbol": self.test_symbol}), content_type="application/json")

        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(response.content)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_duplicate_handling(self, mock_get):
        """Test duplicate article handling"""
        FinancialNewsAlphaV.objects.create(
            title="TSLA earnings report",
            source="Reuters",
            url="https://example.com/news/4",
            summary="Tesla reported Q1 earnings.",
            sentiment_score=0.45,
            sentiment_label="neutral",
            publication_date=datetime(2024, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
            symbol="TSLA"
        )

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "feed": [
                {
                    "title": "New TSLA product announcement",
                    "source": "Business Insider",
                    "url": "https://example.com/news/5",
                    "summary": "Tesla announces new product.",
                    "overall_sentiment_score": "0.75",
                    "overall_sentiment_label": "positive",
                    "time_published": "20240301T120000"
                },
                {
                    "title": "TSLA earnings report",
                    "source": "Reuters",
                    "url": "https://example.com/news/4",
                    "summary": "Tesla reported Q1 earnings.",
                    "overall_sentiment_score": "0.45",
                    "overall_sentiment_label": "neutral",
                    "time_published": "20240301T100000"
                }
            ]
        }

        request = self.factory.post(self.url, {"symbol": "TSLA"}, format='json')
        response = self.view(request)

        self.assertEqual(FinancialNewsAlphaV.objects.count(), 2)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_default_symbol(self, mock_get):
        """Test fetching news when no symbol is provided"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "feed": [{
                "title": "AAPL releases new iPhone",
                "source": "TechCrunch",
                "url": "https://example.com/news/6",
                "summary": "Apple announces latest iPhone.",
                "overall_sentiment_score": "0.80",
                "overall_sentiment_label": "positive",
                "time_published": "20240301T130000"
            }]
        }

        request = self.factory.post(self.url, {}, format='json')
        response = self.view(request)

        self.assertEqual(FinancialNewsAlphaV.objects.first().symbol, self.test_symbol)

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_adage_serializer_failure(self, mock_get):
        """Test serializer failure scenario"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "feed": [{
                "title": "GOOG stock analysis",
                "source": "Seeking Alpha",
                "url": "https://example.com/news/7",
                "summary": "Analysis of Google stock performance.",
                "overall_sentiment_score": "0.60",
                "overall_sentiment_label": "positive",
                "time_published": "20240301T140000"
            }]
        }

        with patch('myapp.Views.financialNewsView.AdageFinancialNewsDatasetSerializer') as mock_serializer:
            mock_serializer_instance = MagicMock()
            mock_serializer_instance.is_valid.return_value = False
            mock_serializer.return_value = mock_serializer_instance

            request = self.factory.post(self.url, {"symbol": "GOOG"}, format='json')
            response = self.view(request)

            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
