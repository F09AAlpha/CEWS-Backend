from django.test import LiveServerTestCase
from django.urls import reverse
import requests
import time
from unittest.mock import patch

from myapp.Models.anomalyDetectionModel import AnomalyDetectionResult


class AnomalyDetectionWorkflowTest(LiveServerTestCase):
    """End-to-end tests for the anomaly detection workflow."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.api_base = cls.live_server_url

    def setUp(self):
        """Set up test environment."""
        # Test request data
        self.test_data = {
            'base': 'USD',
            'target': 'EUR',
            'days': 30
        }

        # Calculate API URL
        self.api_url = f"{self.api_base}{reverse('anomaly-detection')}"

    @patch('myapp.Views.anomalyDetectionView.AnomalyDetectionService.detect_anomalies')
    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_full_anomaly_detection_workflow(self, mock_get_rates, mock_detect_anomalies):
        """
        Test the complete anomaly detection workflow from API request to database storage.

        This test simulates:
        1. User submitting an API request for anomaly detection
        2. System processing the request and detecting anomalies
        3. System returning ADAGE 3.0 formatted response
        4. System storing the results in the database
        """
        # Mock the anomaly detection service to return a known result
        mock_anomaly_result = {
            'base': 'USD',
            'target': 'EUR',
            'anomaly_count': 2,
            'analysis_period_days': 30,
            'anomaly_points': [
                {
                    'timestamp': '2023-03-15T00:00:00.000Z',
                    'rate': 0.95,
                    'z_score': 2.5,
                    'percent_change': 1.8
                },
                {
                    'timestamp': '2023-03-20T00:00:00.000Z',
                    'rate': 0.91,
                    'z_score': -2.2,
                    'percent_change': -1.5
                }
            ]
        }
        mock_detect_anomalies.return_value = mock_anomaly_result

        # Step 1: Make the API request
        response = requests.post(
            self.api_url,
            json=self.test_data,
            headers={'Content-Type': 'application/json'}
        )

        # Step 2: Verify successful response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Step 3: Verify ADAGE 3.0 format in response
        self.assertEqual(data['data_source'], 'Alpha Vantage')
        self.assertIn('dataset_id', data)
        self.assertIn('time_object', data)
        self.assertIn('events', data)

        # Verify events match our mock data
        events = data['events']
        self.assertEqual(len(events), 2)

        # Step 4: Verify database storage (allowing some time for async operations if needed)
        time.sleep(0.5)  # Small delay to ensure DB operations complete

        # Verify a record was created in the database
        results = AnomalyDetectionResult.objects.filter(
            base_currency='USD',
            target_currency='EUR'
        )
        self.assertEqual(results.count(), 1)

        # Verify result details
        result = results.first()
        self.assertEqual(result.anomaly_count, 2)
        self.assertEqual(result.analysis_period_days, 30)

        # Verify JSON data storage
        result_data = result.result_data
        self.assertIsNotNone(result_data)
        self.assertEqual(result_data['data_source'], 'Alpha Vantage')
        self.assertGreater(len(result_data['events']), 0)

    @patch('myapp.Service.alpha_vantage.AlphaVantageService.get_exchange_rates')
    def test_error_workflow(self, mock_get_rates):
        """Test the workflow when an error occurs during processing."""
        # Mock an API error
        mock_get_rates.side_effect = Exception("API service unavailable")

        # Make the API request
        response = requests.post(
            self.api_url,
            json=self.test_data,
            headers={'Content-Type': 'application/json'}
        )

        # Verify error response
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn('detail', data)

        # Verify no records were created in the database
        results = AnomalyDetectionResult.objects.filter(
            base_currency='USD',
            target_currency='EUR'
        )
        self.assertEqual(results.count(), 0)

    def test_invalid_input_workflow(self):
        """Test the workflow with invalid input data."""
        # Invalid data: missing target currency
        invalid_data = {'base': 'USD'}

        # Make the API request
        response = requests.post(
            self.api_url,
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )

        # Verify error response
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('detail', data)

        # Try another invalid case: same currencies
        invalid_data = {'base': 'USD', 'target': 'USD'}

        # Make the API request
        response = requests.post(
            self.api_url,
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )

        # Verify error response
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('detail', data)
