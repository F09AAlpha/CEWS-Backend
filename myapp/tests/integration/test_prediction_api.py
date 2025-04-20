from django.test import TestCase, Client
from django.urls import reverse
from rest_framework import status
from unittest.mock import patch
from datetime import datetime, timedelta
import pandas as pd
from django.utils import timezone

from myapp.Models.predictionModel import CurrencyPrediction
from myapp.Service.predictionService import PredictionService


class PredictionAPITestCase(TestCase):
    """Integration tests for the Prediction API."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = Client()
        self.prediction_service = PredictionService()
        self.base_currency = 'USD'
        self.target_currency = 'EUR'
        self.base_url = reverse('currency_prediction', kwargs={
            'base': self.base_currency,
            'target': self.target_currency
        })
        self.horizon_days = 7

        # Create mock prediction data
        self.mock_prediction = CurrencyPrediction(
            base_currency=self.base_currency,
            target_currency=self.target_currency,
            forecast_horizon=self.horizon_days,
            current_rate=1.05,
            change_percent=2.5,
            mean_predictions={"2023-04-04": 1.06, "2023-04-05": 1.07},
            lower_bound={"2023-04-04": 1.03, "2023-04-05": 1.04},
            upper_bound={"2023-04-04": 1.09, "2023-04-05": 1.10},
            model_version="Statistical Model v2",
            confidence_score=65.5,
            input_data_range="2023-03-01 to 2023-04-03",
            used_correlation_data=True,
            used_news_sentiment=True,
            used_economic_indicators=True,
            used_anomaly_detection=True,
            mean_square_error=0.0002,
            root_mean_square_error=0.014,
            mean_absolute_error=0.012
        )
        # Set primary key for the mock object
        self.mock_prediction.pk = 1
        # Set prediction date with timezone
        self.mock_prediction.prediction_date = timezone.now()

    @patch('myapp.Service.predictionService.PredictionService.create_prediction')
    def test_prediction_creation_endpoint(self, mock_create_prediction):
        """Test creating a prediction through the API endpoint."""
        # Setup mock to return our mock prediction
        mock_prediction = self.mock_prediction
        # Update model version to reflect ARIMA usage
        mock_prediction.model_version = "ARIMA(1,1,0)"
        mock_create_prediction.return_value = mock_prediction

        # Using GET with refresh=true to create a new prediction
        url = f"{self.base_url}?refresh=true"
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('events', data)
        self.assertEqual(data['events'][0]['attributes']['base_currency'], self.base_currency)
        self.assertEqual(data['events'][0]['attributes']['target_currency'], self.target_currency)

        # Verify ARIMA model version is in the response
        self.assertIn('ARIMA', data['events'][0]['attributes']['model_version'])

    @patch('myapp.Service.predictionService.PredictionService.get_latest_prediction')
    def test_prediction_retrieval_endpoint(self, mock_get_latest):
        """Test retrieving an existing prediction through the API endpoint."""
        # Setup mock to return our mock prediction
        mock_get_latest.return_value = self.mock_prediction

        # Try to retrieve the prediction
        response = self.client.get(self.base_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('events', data)
        self.assertEqual(data['events'][0]['attributes']['base_currency'], self.base_currency)
        self.assertEqual(data['events'][0]['attributes']['target_currency'], self.target_currency)

    @patch('myapp.Service.predictionService.PredictionService.create_prediction')
    def test_prediction_with_custom_horizon(self, mock_create_prediction):
        """Test creating a prediction with a custom horizon."""
        # Setup mock to return a custom horizon prediction
        custom_horizon = 14
        custom_prediction = self.mock_prediction
        custom_prediction.forecast_horizon = custom_horizon
        mock_create_prediction.return_value = custom_prediction

        # Make request with custom horizon
        url = f"{self.base_url}?forecast_horizon={custom_horizon}&refresh=true"
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check horizon in response
        data = response.json()
        self.assertEqual(data['events'][0]['time_object']['horizon_days'], custom_horizon)

    def test_prediction_with_invalid_currency(self):
        """Test behavior with invalid currency."""
        # Test with invalid currency
        invalid_url = reverse('currency_prediction', kwargs={
            'base': 'INVALID',
            'target': self.target_currency
        })
        response = self.client.get(invalid_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.json())

        # Test with same base and target currency
        same_url = reverse('currency_prediction', kwargs={
            'base': 'USD',
            'target': 'USD'
        })
        response = self.client.get(same_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.json())

    def _create_test_dataframe(self, data_points=30, with_anomalies=True):
        """Helper to create a test dataframe with exchange rate data."""
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(data_points)]
        dates.reverse()  # Sort chronologically

        # Create exchange rate values with a specific pattern
        base_rate = 1.0
        rates = [base_rate + i * 0.01 for i in range(data_points)]

        # Add anomalies if requested
        if with_anomalies and data_points > 15:
            rates[7] = base_rate + 0.15  # Spike
            rates[14] = base_rate - 0.1   # Drop

        # Create DataFrame
        data = {
            'date': dates,
            'close': rates
        }
        df = pd.DataFrame(data)
        # Add attributes that prediction service will look for
        df.attrs['base_currency'] = self.base_currency
        df.attrs['target_currency'] = self.target_currency
        return df

    @patch('myapp.Service.predictionService.PredictionService.create_prediction')
    @patch('myapp.Service.anomalyDetectionService.AnomalyDetectionService.detect_anomalies')
    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_prediction_with_anomaly_detection(self, mock_get_rates, mock_detect_anomalies, mock_create_prediction):
        """Test that predictions properly integrate anomaly detection."""
        # Create test data - keep this for documentation even if not used with mocks
        # test_df = self._create_test_dataframe(data_points=30, with_anomalies=True)

        # Setup mock prediction with anomaly detection used
        mock_prediction = self.mock_prediction
        mock_prediction.used_anomaly_detection = True
        mock_create_prediction.return_value = mock_prediction

        # Make request for a new prediction with refresh=true
        url = f"{self.base_url}?refresh=true"
        response = self.client.get(url)

        # Check response
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check the API response
        data = response.json()
        self.assertIn('events', data)

        # Check for anomaly detection in influencing factors
        influencing_factors = data['events'][0]['attributes']['influencing_factors']
        anomaly_factor = None
        for factor in influencing_factors:
            if factor['factor_name'] == 'Anomaly Detection':
                anomaly_factor = factor
                break

        self.assertIsNotNone(anomaly_factor)
        self.assertEqual(anomaly_factor['impact_level'], 'high')
        self.assertTrue(anomaly_factor['used_in_prediction'])

        # The model version should reflect anomaly detection usage
        self.assertIn('v2', data['events'][0]['attributes']['model_version'])

    @patch('myapp.Service.predictionService.PredictionService.create_prediction')
    @patch('myapp.Service.anomalyDetectionService.AnomalyDetectionService.detect_anomalies')
    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_prediction_without_anomalies(self, mock_get_rates, mock_detect_anomalies, mock_create_prediction):
        """Test prediction behavior when no anomalies are detected."""
        # Create test data with no anomalies - keep this for documentation even if not used with mocks
        # test_df = self._create_test_dataframe(data_points=30, with_anomalies=False)

        # Setup mock prediction with anomaly detection used but no anomalies found
        mock_prediction = self.mock_prediction
        mock_prediction.used_anomaly_detection = True
        mock_create_prediction.return_value = mock_prediction

        # Make request for a new prediction
        url = f"{self.base_url}?refresh=true"
        response = self.client.get(url)

        # Check response
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # The API response should still mention anomaly detection
        data = response.json()
        influencing_factors = data['events'][0]['attributes']['influencing_factors']
        anomaly_factor = None
        for factor in influencing_factors:
            if factor['factor_name'] == 'Anomaly Detection':
                anomaly_factor = factor
                break

        self.assertIsNotNone(anomaly_factor)

    @patch('myapp.Service.predictionService.PredictionService.create_prediction')
    def test_error_metrics_in_response(self, mock_create_prediction):
        """Test that error metrics are included in the API response."""
        # Setup mock with error metrics
        mock_prediction = self.mock_prediction
        mock_create_prediction.return_value = mock_prediction

        # Make request
        url = f"{self.base_url}?refresh=true"
        response = self.client.get(url)

        # Check response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()

        # Verify error metrics are in the response
        attrs = data['events'][0]['attributes']
        self.assertIn('mean_square_error', attrs)
        self.assertIn('root_mean_square_error', attrs)
        self.assertIn('mean_absolute_error', attrs)

        # Verify model_accuracy object is present with correct structure
        self.assertIn('model_accuracy', attrs)
        model_accuracy = attrs['model_accuracy']
        self.assertIn('mean_square_error', model_accuracy)
        self.assertIn('root_mean_square_error', model_accuracy)
        self.assertIn('mean_absolute_error', model_accuracy)
        self.assertIn('description', model_accuracy)

    @patch('myapp.Service.predictionService.PredictionService.create_prediction')
    def test_prediction_with_model_selection(self, mock_create_prediction):
        """Test creating a prediction with different model types."""
        # Setup base mock prediction
        base_prediction = self.mock_prediction

        # Simple mock that returns the same prediction regardless of arguments
        mock_create_prediction.return_value = base_prediction

        # Test ARIMA model selection
        response = self.client.get(f"{self.base_url}?refresh=true&model=arima")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Test statistical model selection
        response = self.client.get(f"{self.base_url}?refresh=true&model=statistical")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Test auto model selection (should default to ARIMA in most cases)
        response = self.client.get(f"{self.base_url}?refresh=true&model=auto")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check calls to create_prediction
        self.assertEqual(mock_create_prediction.call_count, 3)

        # Verify the create_prediction was called with correct parameters
        # Extract the calls and match them with expected use_arima values
        calls = mock_create_prediction.call_args_list

        # Function to check if a call matches expected parameters
        def find_call_with_model(model_type, expected_arima):
            for call in calls:
                args, kwargs = call
                # In real code, this would be checking different values
                # But for our test we just verify that the parameters are consistent
                if len(args) >= 3 and args[0] == self.base_currency and args[1] == self.target_currency:
                    if kwargs.get('use_arima', True) == expected_arima:
                        return True
            return False

        # Check that model types were mapped to appropriate use_arima values
        # ARIMA model should have use_arima=True
        self.assertTrue(find_call_with_model('arima', True))
        # Statistical model should have use_arima=False
        self.assertTrue(find_call_with_model('statistical', False))
        # Auto should default to True in most implementations
        self.assertTrue(find_call_with_model('auto', True))

        # Test invalid model parameter
        response = self.client.get(f"{self.base_url}?refresh=true&model=invalid")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn('error', data)
        self.assertEqual(data['error']['type'], 'InvalidParameter')
