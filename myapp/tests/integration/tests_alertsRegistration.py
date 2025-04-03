import json
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert

class RegisterAlertIntegrationTest(APITestCase):
    """Integration tests for the alert registration API"""

    def setUp(self):
        """Set up test environment"""
        self.url = reverse('register-alert')
        self.valid_data = {
            "base": "USD",
            "target": "EUR",
            "alert_type": "above",
            "threshold": "1.10",
            "email": "test@example.com"
        }

    def test_register_alert_success(self):
        """Test successful alert registration"""
        response = self.client.post(
            self.url, data=json.dumps(self.valid_data), content_type="application/json"
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("alert_id", response.data)
        self.assertEqual(response.data["status"], "registered")

    def test_register_alert_missing_data(self):
        """Test registration failure due to missing required fields"""
        response = self.client.post(self.url, json.dumps({}), content_type="application/json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("base", response.data)  # Ensure error message for missing field

    def test_register_alert_invalid_email(self):
        """Test registration failure due to invalid email"""
        invalid_data = self.valid_data.copy()
        invalid_data["email"] = "invalid-email"
        response = self.client.post(self.url, json.dumps(invalid_data), content_type="application/json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("email", response.data)
