from django.test import TestCase
from myapp.Models.financialNewsModel import FinancialNews
from myapp.Views.financialNewsView import FetchFinancialNewsView
from rest_framework.test import APIRequestFactory
from unittest.mock import patch


class FinancialNewsFetchTest(TestCase):
    
    def setUp(self):
        self.factory = APIRequestFactory()
        self.view = FetchFinancialNewsView.as_view()

    @patch("myapp.Views.financialNewsView.requests.get")
    def test_fetch_news_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "articles": [
                {
                    "title": "Test Article",
                    "source": {"name": "Forbes"},
                    "url": "https://forbes.com/test-article",
                    "publishedAt": "2025-02-15T12:00:00Z"
                }
            ]
        }

        request = self.factory.get("/fetch-news/")
        response = self.view(request)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(FinancialNews.objects.count(), 1)
