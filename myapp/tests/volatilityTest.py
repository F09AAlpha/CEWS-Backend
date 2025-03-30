from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch


class VolatilityAnalysisTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_forex_daily')
    def test_volatility_analysis(self, mock_get_forex_daily):
        # Create mock data
        dates = pd.date_range(start=datetime.now().date().isoformat(), periods=50, freq='-1D')
        mock_data = pd.DataFrame({
            'open': np.random.normal(1.2, 0.01, 50),
            'high': np.random.normal(1.21, 0.01, 50),
            'low': np.random.normal(1.19, 0.01, 50),
            'close': np.random.normal(1.2, 0.01, 50)
        }, index=dates)

        mock_get_forex_daily.return_value = mock_data

        # Test the API endpoint
        url = reverse('volatility_analysis', kwargs={'base': 'EUR', 'target': 'USD'})
        response = self.client.get(url)

        # Check response
        self.assertEqual(response.status_code, 200)

        # Verify the response structure
        data = response.json()
        self.assertEqual(data['base'], 'EUR')
        self.assertEqual(data['target'], 'USD')
        self.assertIn('current_volatility', data)
        self.assertIn('average_volatility', data)
        self.assertIn('volatility_level', data)
        self.assertIn('analysis_period_days', data)
        self.assertIn('trend', data)
