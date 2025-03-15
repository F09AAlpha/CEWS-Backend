from rest_framework.test import APITestCase
from rest_framework import status
from myapp.Models.financialNewsModel import FinancialNews
from unittest.mock import patch

class FetchFinancialNewsViewTest(APITestCase):
    def setUp(self):
        # Setup URL to be used for testing
        self.url = '/api/fetch-financial-news/'

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_success(self, mock_get):
        # Define the mock response structure based on the API
        mock_response = {
            "status": "ok",
            "articles": [
                {
                    "title": "Sample Financial News",
                    "source": {"name": "Sample Source"},
                    "url": "https://example.com/news/1",
                    "publishedAt": "2025-03-14T00:00:00Z"
                },
                {
                    "title": "Another Financial News",
                    "source": {"name": "Another Source"},
                    "url": "https://example.com/news/2",
                    "publishedAt": "2025-03-14T01:00:00Z"
                }
            ]
        }

        # Mock the external API's GET request to return the mock response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Send GET request to the FetchFinancialNewsView
        response = self.client.get(self.url)

        # Assert the status code returned is 201, as per your view's behavior
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Check if the data was saved in the database
        self.assertEqual(FinancialNews.objects.count(), 2)  # Expect 2 articles to be saved

        # Check the first stored news article
        news_1 = FinancialNews.objects.get(url="https://example.com/news/1")
        self.assertEqual(news_1.title, "Sample Financial News")
        self.assertEqual(news_1.source, "Sample Source")
        self.assertEqual(news_1.url, "https://example.com/news/1")
        
        # Check the second stored news article
        news_2 = FinancialNews.objects.get(url="https://example.com/news/2")
        self.assertEqual(news_2.title, "Another Financial News")
        self.assertEqual(news_2.source, "Another Source")
        self.assertEqual(news_2.url, "https://example.com/news/2")

    @patch('myapp.Views.financialNewsView.requests.get')
    def test_fetch_financial_news_failure(self, mock_get):
        # Mock a failed API request
        mock_get.return_value.status_code = 500
        mock_get.return_value.json.return_value = {"status": "error", "message": "Internal Server Error"}

        # Send GET request to your view
        response = self.client.get(self.url)

        # Assert the status code is 500 as per the error response from the external API
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("error", response.data)
