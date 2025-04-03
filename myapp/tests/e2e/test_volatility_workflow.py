from django.test import LiveServerTestCase
from django.urls import reverse
import requests
from unittest.mock import patch


class VolatilityWorkflowTest(LiveServerTestCase):
    """End-to-end tests for the volatility analysis workflow."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.api_base = cls.live_server_url

    def setUp(self):
        """Set up test environment."""
        # Test data
        self.base_currency = 'USD'
        self.target_currency = 'EUR'
        self.days = 30

        # Calculate API URL with path parameters
        kwargs = {'base': self.base_currency, 'target': self.target_currency}
        base_path = reverse('volatility_analysis', kwargs=kwargs)
        self.api_url = f"{self.api_base}{base_path}"

    @patch('myapp.Service.volatilityService.VolatilityService.calculate_volatility')
    def test_full_volatility_workflow(self, mock_calculate_volatility):
        """
        Test the complete volatility analysis workflow from API request to response.

        This test simulates:
        1. User requesting volatility analysis for a currency pair
        2. System processing the request and calculating volatility metrics
        3. System returning ADAGE 3.0 formatted response with volatility metrics
        """
        # Mock the volatility calculation with a known result
        mock_result = {
            'data_source': 'Alpha Vantage',
            'dataset_type': 'Currency Volatility Analysis',
            'dataset_id': f"volatility_analysis_{self.base_currency}_{self.target_currency}_20230401",
            'time_object': {
                'timestamp': '2023-04-01T12:00:00.000Z',
                'timezone': 'GMT+0'
            },
            'events': [
                {
                    'time_object': {
                        'timestamp': '2023-04-01T12:00:00.000Z',
                        'duration': self.days,
                        'duration_unit': 'day',
                        'timezone': 'GMT+0'
                    },
                    'event_type': 'volatility_analysis',
                    'attributes': {
                        'base_currency': self.base_currency,
                        'target_currency': self.target_currency,
                        'current_volatility': 8.75,
                        'average_volatility': 7.92,
                        'volatility_level': 'NORMAL',
                        'trend': 'STABLE',
                        'data_points': 30,
                        'confidence_score': 100.0,
                        'analysis_period_days': self.days
                    }
                }
            ]
        }
        mock_calculate_volatility.return_value = mock_result

        # Step 1: Make the API request
        response = requests.get(self.api_url)

        # Step 2: Verify successful response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Step 3: Verify ADAGE 3.0 format
        self.assertEqual(data['data_source'], 'Alpha Vantage')
        self.assertEqual(data['dataset_type'], 'Currency Volatility Analysis')
        self.assertIn('dataset_id', data)
        self.assertIn('time_object', data)
        self.assertIn('events', data)

        # Verify events match our mock data
        events = data['events']
        self.assertEqual(len(events), 1)

        # Verify event content
        event = events[0]
        self.assertEqual(event['event_type'], 'volatility_analysis')
        self.assertEqual(event['attributes']['base_currency'], self.base_currency)
        self.assertEqual(event['attributes']['target_currency'], self.target_currency)
        self.assertEqual(event['attributes']['volatility_level'], 'NORMAL')
        self.assertEqual(event['attributes']['trend'], 'STABLE')

        # Verify headers
        self.assertEqual(response.headers.get('X-ADAGE-Version'), '3.0')

    def test_volatility_workflow_with_parameters(self):
        """
        Test the workflow with query parameters.

        Tests that the days parameter is properly processed from request to result.
        """
        # Make request with days parameter
        custom_days = 90
        response = requests.get(f"{self.api_url}?days={custom_days}")

        # Verify successful response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify days parameter is reflected in result
        event = data['events'][0]
        self.assertEqual(event['time_object']['duration'], custom_days)
        self.assertEqual(event['attributes']['analysis_period_days'], custom_days)

    def test_volatility_workflow_invalid_currencies(self):
        """
        Test the workflow with invalid currency codes.

        Tests error handling for invalid inputs.
        """
        # Test with invalid base currency
        kwargs = {'base': 'INVALID', 'target': 'EUR'}
        invalid_base_path = reverse('volatility_analysis', kwargs=kwargs)
        invalid_base_url = f"{self.api_base}{invalid_base_path}"
        response_invalid_base = requests.get(invalid_base_url)

        # Verify error response - should now be 400 Bad Request
        self.assertEqual(response_invalid_base.status_code, 400)
        data = response_invalid_base.json()
        self.assertIn('error', data)
        self.assertEqual(data['error']['type'], 'InvalidCurrencyError')
        self.assertIn('Invalid base currency code', data['error']['message'])

        # Test with invalid target currency
        kwargs = {'base': 'USD', 'target': 'INVALID'}
        invalid_target_path = reverse('volatility_analysis', kwargs=kwargs)
        invalid_target_url = f"{self.api_base}{invalid_target_path}"
        response_invalid_target = requests.get(invalid_target_url)

        # Verify error response - should now be 400 Bad Request
        self.assertEqual(response_invalid_target.status_code, 400)
        data = response_invalid_target.json()
        self.assertIn('error', data)
        self.assertEqual(data['error']['type'], 'InvalidCurrencyError')
        self.assertIn('Invalid target currency code', data['error']['message'])

    def test_volatility_workflow_error_handling(self):
        """
        Test the workflow with error conditions.

        Tests proper error handling when an invalid request is made.
        """
        # Use an invalid URL that should trigger a 404 error
        error_url = f"{self.api_base}/invalid/url/that/should/not/exist"

        # Make request
        response = requests.get(error_url)

        # Verify error response is 404
        self.assertEqual(response.status_code, 404)

        # Make a request with an invalid parameter type to cause a validation error
        invalid_param_url = f"{self.api_url}?days=not_a_number"
        response_invalid = requests.get(invalid_param_url)

        # This should return a 400 Bad Request or 500 Internal Server Error
        self.assertIn(response_invalid.status_code, [400, 500])
