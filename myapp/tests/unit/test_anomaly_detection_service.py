import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

from myapp.Service.anomalyDetectionService import (
    AnomalyDetectionService,
    ProcessingError
)


class TestAnomalyDetectionService(unittest.TestCase):
    """Test cases for the AnomalyDetectionService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock Alpha Vantage service
        self.alpha_vantage_mock = MagicMock()

        # Create the service with the mock
        self.service = AnomalyDetectionService(
            base_currency="USD",
            target_currency="EUR",
            analysis_period_days=30,
            z_score_threshold=2.0,
            alpha_vantage_service=self.alpha_vantage_mock
        )

    def test_init(self):
        """Test service initialization."""
        self.assertEqual(self.service.base_currency, "USD")
        self.assertEqual(self.service.target_currency, "EUR")
        self.assertEqual(self.service.analysis_period_days, 30)
        self.assertEqual(self.service.z_score_threshold, 2.0)
        self.assertEqual(self.service.alpha_vantage, self.alpha_vantage_mock)

    def _create_test_dataframe(self, data_points=30, with_anomalies=True):
        """Helper to create a test DataFrame."""
        # Create dates for the specified number of days
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(data_points)]
        dates.reverse()  # Sort chronologically

        # Create exchange rate values with a specific pattern
        base_rate = 1.0
        rates = [base_rate + i * 0.01 for i in range(data_points)]  # Slight upward trend

        # Add anomalies if requested
        if with_anomalies:
            # Add an anomaly at 1/4 of the way through
            anomaly_index = data_points // 4
            rates[anomaly_index] = base_rate + 0.15  # Clear spike

            # Add another anomaly at 3/4 of the way through
            anomaly_index = 3 * data_points // 4
            rates[anomaly_index] = base_rate - 0.1  # Clear drop

        # Create DataFrame
        data = {
            'date': dates,
            'close': rates
        }
        return pd.DataFrame(data)

    def test_get_exchange_rates(self):
        """Test retrieving exchange rates."""
        # Mock the Alpha Vantage service response
        test_df = self._create_test_dataframe()
        self.alpha_vantage_mock.get_exchange_rates.return_value = test_df

        # Call the method
        result = self.service.get_exchange_rates()

        # Verify the service called Alpha Vantage correctly
        self.alpha_vantage_mock.get_exchange_rates.assert_called_once_with(
            "USD", "EUR", days=30
        )

        # Verify the result has the expected columns
        self.assertIn('date', result.columns)
        self.assertIn('rate', result.columns)
        self.assertEqual(len(result), 30)

    def test_get_exchange_rates_insufficient_data(self):
        """Test handling insufficient data."""
        # Mock Alpha Vantage returning very little data
        test_df = self._create_test_dataframe(data_points=5)
        self.alpha_vantage_mock.get_exchange_rates.return_value = test_df

        # Expect ProcessingError when calling the method
        with self.assertRaises(ProcessingError) as context:
            self.service.get_exchange_rates()

        # Verify the error message contains information about insufficient data
        self.assertIn("Not enough data for analysis", str(context.exception))

    def test_detect_anomalies(self):
        """Test detecting anomalies in exchange rate data."""
        # Mock the exchange rates with known anomalies
        test_df = self._create_test_dataframe(with_anomalies=True)
        self.alpha_vantage_mock.get_exchange_rates.return_value = test_df

        # Call the method
        result = self.service.detect_anomalies()

        # Verify result structure
        self.assertEqual(result['base'], "USD")
        self.assertEqual(result['target'], "EUR")
        self.assertIn('anomaly_count', result)
        self.assertIn('anomaly_points', result)
        self.assertGreater(result['anomaly_count'], 0)

        # Verify anomaly points have the expected structure
        for point in result['anomaly_points']:
            self.assertIn('timestamp', point)
            self.assertIn('rate', point)
            self.assertIn('z_score', point)
            self.assertIn('percent_change', point)
            # Anomalies should have z-scores above the threshold
            self.assertGreater(abs(point['z_score']), 2.0)

    def test_detect_anomalies_no_anomalies(self):
        """Test detecting no anomalies in stable exchange rate data."""
        # Mock exchange rates with no anomalies
        test_df = self._create_test_dataframe(with_anomalies=False)
        self.alpha_vantage_mock.get_exchange_rates.return_value = test_df

        # Call the method
        result = self.service.detect_anomalies()

        # Verify result
        self.assertEqual(result['anomaly_count'], 0)
        self.assertEqual(len(result['anomaly_points']), 0)

    @patch('myapp.Service.anomalyDetectionService.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in the service."""
        # Mock Alpha Vantage service to raise an exception
        self.alpha_vantage_mock.get_exchange_rates.side_effect = Exception("API error")

        # Expect exception to be wrapped
        with self.assertRaises(ProcessingError):
            self.service.detect_anomalies()

        # Verify logger was called
        mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main()
