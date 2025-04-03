import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from myapp.Service.predictionService import PredictionService


class TestPredictionService(unittest.TestCase):
    """Test cases for the PredictionService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create the service
        self.service = PredictionService()

        # Mock the alpha_vantage_service attribute
        self.service.alpha_vantage_service = MagicMock()

        # Mock the correlation_service attribute
        self.service.correlation_service = MagicMock()

        # Create test exchange rate data
        self.exchange_df = self._create_exchange_dataframe()

    def _create_exchange_dataframe(self, days=30):
        """Helper to create a test exchange rate DataFrame."""
        # Create date range
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).date() for i in range(days)]
        dates.reverse()  # Sort chronologically

        # Create price data with some trend and volatility
        np.random.seed(42)  # For reproducibility

        # Create base price series with trend (slight upward)
        base_price = 1.0
        trend = np.linspace(0, 0.05, days)  # Slight upward trend
        noise = np.random.normal(0, 0.01, days)  # Daily noise
        prices = base_price + trend + np.cumsum(noise)  # Cumulative noise for random walk

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)  # Add 0 for first day

        # Create dataframe
        data = {
            'date': dates,
            'close': prices,
            'open': prices - 0.01,
            'high': prices + 0.02,
            'low': prices - 0.02,
            'return': returns
        }

        return pd.DataFrame(data)

    def test_init(self):
        """Test service initialization."""
        service = PredictionService()
        self.assertIsNotNone(service.alpha_vantage_service)
        self.assertIsNotNone(service.correlation_service)

    def test_predict_with_statistical_model(self):
        """Test the statistical prediction method."""
        # Setup exchange dataframe
        exchange_df = self.exchange_df.copy()

        # Call the method
        prediction_results = self.service._predict_with_statistical_model(
            exchange_df, None, 7
        )

        # Verify result structure
        self.assertIn('mean_predictions', prediction_results)
        self.assertIn('lower_bound', prediction_results)
        self.assertIn('upper_bound', prediction_results)
        self.assertIn('model_version', prediction_results)
        self.assertIn('confidence_score', prediction_results)
        self.assertEqual(len(prediction_results['mean_predictions']), 7)

        # Verify values are numeric
        for date, value in prediction_results['mean_predictions'].items():
            self.assertIsInstance(value, float)

        # Verify lower bound is less than mean is less than upper bound
        for date in prediction_results['mean_predictions'].keys():
            self.assertLess(prediction_results['lower_bound'][date], prediction_results['mean_predictions'][date])
            self.assertLess(prediction_results['mean_predictions'][date], prediction_results['upper_bound'][date])

    @patch('myapp.Models.predictionModel.CurrencyPrediction.objects')
    def test_get_latest_prediction(self, mock_objects):
        """Test getting the latest prediction."""
        # Setup mock
        mock_prediction = MagicMock()
        mock_objects.filter.return_value.order_by.return_value.first.return_value = mock_prediction

        # Call the method
        result = self.service.get_latest_prediction('USD', 'EUR')

        # Verify result
        self.assertEqual(result, mock_prediction)
        mock_objects.filter.assert_called_once_with(base_currency='USD', target_currency='EUR')

    @patch('myapp.Service.predictionService.PredictionService.get_latest_prediction')
    @patch('myapp.Service.predictionService.PredictionService._predict_with_statistical_model')
    def test_create_prediction(self, mock_predict, mock_get_latest):
        """Test creating a new prediction."""
        # Setup mocks
        mock_get_latest.return_value = None
        exchange_df = self.exchange_df.copy()
        self.service.alpha_vantage_service.get_exchange_rates.return_value = exchange_df

        prediction_data = {
            'mean_predictions': {'2023-01-01': 1.1},
            'lower_bound': {'2023-01-01': 1.0},
            'upper_bound': {'2023-01-01': 1.2},
            'model_version': 'Statistical Model v2',
            'confidence_score': 80.5,
            'input_data_range': '2022-12-01 to 2022-12-31',
            'used_correlation_data': False,
            'used_news_sentiment': False,
            'used_economic_indicators': False,
            'used_anomaly_detection': False
        }
        mock_predict.return_value = prediction_data

        # Call the method
        result = self.service.create_prediction('USD', 'EUR', 7, refresh=True)

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.base_currency, 'USD')
        self.assertEqual(result.target_currency, 'EUR')
        self.assertEqual(result.forecast_horizon, 7)
        self.assertEqual(result.mean_predictions, prediction_data['mean_predictions'])
        mock_predict.assert_called_once()


if __name__ == '__main__':
    unittest.main()
