from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch
from rest_framework import status
from myapp.Models.currencyNewsModel import CurrencyNewsAlphaV
from myapp.Models.financialNewsModel import FinancialNewsAlphaV
from myapp.Serializers.currencyNewsSerializer import CurrencyNewsSerializer
from myapp.Serializers.financialNewsSerializer import FinancialNewsSerializer
from datetime import datetime, timezone
import uuid

class NewsAPITestCase(TestCase):
    def setUp(self):
        self.currency_news_url = reverse("currency-news-list")
        self.financial_news_url = reverse("fetch-financial-news")
        self.currency = "USD"
        self.symbol = "AAPL"

        self.mock_currency_news = {
            "title": "USD Market Updates",
            "source": "Bloomberg",
            "url": "https://example.com/news1",
            "summary": "USD gains strength amid market shifts.",
            "sentiment_score": 0.5,
            "sentiment_label": "positive",
            "publication_date": datetime.now(timezone.utc),
            "currency": self.currency
        }
        self.mock_financial_news = {
            "title": "AAPL Stock Surges",
            "source": "Reuters",
            "url": "https://example.com/news2",
            "summary": "Apple stock sees record high.",
            "sentiment_score": 0.7,
            "sentiment_label": "positive",
            "publication_date": datetime.now(timezone.utc),
            "symbol": self.symbol
        }

    @patch("requests.get")
    def test_fetch_currency_news_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": [self.mock_currency_news]}
        
        response = self.client.post(f"/v1/news/events?currency={self.currency}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("dataset_id", response.data)

    @patch("requests.get")
    def test_fetch_financial_news_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": [self.mock_financial_news]}
        
        response = self.client.post("/v1/financial/", {"symbol": self.symbol})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("dataset_id", response.data)

    @patch("requests.get")
    def test_fetch_currency_news_no_results(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": []}
        
        response = self.client.post(f"/v1/news/events?currency={self.currency}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], f"No currency news data found for {self.currency}")

    @patch("requests.get")
    def test_fetch_financial_news_no_results(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": []}
        
        response = self.client.post("/v1/financial/", {"symbol": self.symbol})
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.data["error"], "No financial news data found in the API response.")

    @patch("requests.get")
    def test_fetch_currency_news_api_failure(self, mock_get):
        mock_get.return_value.status_code = 500
        response = self.client.post(f"/v1/news/events?currency={self.currency}")
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    @patch("requests.get")
    def test_fetch_financial_news_api_failure(self, mock_get):
        mock_get.return_value.status_code = 500
        response = self.client.post("/v1/financial/", {"symbol": self.symbol})
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def test_currency_news_list_empty(self):
        response = self.client.get(self.currency_news_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, [])

    def test_financial_news_list_empty(self):
        response = self.client.get(self.financial_news_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, [])

    def test_invalid_currency_parameter(self):
        response = self.client.post("/v1/news/events?currency=123")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_invalid_symbol_parameter(self):
        response = self.client.post("/v1/financial/", {"symbol": "!!@#$$"})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch("requests.get")
    def test_fetch_currency_news_duplicate_entry(self, mock_get):
        CurrencyNewsAlphaV.objects.create(**self.mock_currency_news)
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": [self.mock_currency_news]}
        
        response = self.client.post(f"/v1/news/events?currency={self.currency}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], f"Currency news data fetched and stored for {self.currency}")
        self.assertEqual(CurrencyNewsAlphaV.objects.count(), 1)

    @patch("requests.get")
    def test_fetch_financial_news_duplicate_entry(self, mock_get):
        FinancialNewsAlphaV.objects.create(**self.mock_financial_news)
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"feed": [self.mock_financial_news]}
        
        response = self.client.post("/v1/financial/", {"symbol": self.symbol})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["message"], "Financial news data fetched and stored")
        self.assertEqual(FinancialNewsAlphaV.objects.count(), 1)

    def test_missing_symbol_parameter(self):
        response = self.client.post("/v1/financial/", {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["error"], "Symbol parameter is required.")
