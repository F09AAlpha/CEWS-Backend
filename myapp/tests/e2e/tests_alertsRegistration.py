import json
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert


class RegisterAlertE2ETest(TestCase):

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

    def test_register_alert_e2e(self):
        """Test full alert registration process"""

        # Step 1: Register the alert
        response = self.client.post(
            self.url, json.dumps(self.valid_data), content_type="application/json"
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("alert_id", response.data)

        # Step 2: Check if the alert is stored in the database
        stored_alerts = ExchangeRateAlert.objects.filter(email="test@example.com")
        self.assertEqual(stored_alerts.count(), 1)

        # Step 3: Validate stored alert data
        alert = stored_alerts.first()
        self.assertEqual(alert.base, "USD")
        self.assertEqual(alert.target, "EUR")
        self.assertEqual(alert.alert_type, "above")
