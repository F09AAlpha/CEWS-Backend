from unittest.mock import patch, MagicMock, ANY
from rest_framework import status
from myapp.models import ExchangeRateAlert
from django.test import TestCase
from django.urls import reverse
from rest_framework.exceptions import ErrorDetail


class RegisterAlertViewTest(TestCase):

    @patch("myapp.Serializers.exchangeRateAlertSerializer")
    def test_register_alert_success(self, mock_serializer_class):
        mock_serializer = MagicMock()
        mock_serializer.is_valid.return_value = True
        mock_serializer.save.return_value = ExchangeRateAlert(
            alert_id="ALERT-12345",
            base="USD",
            target="EUR",
            alert_type="above",
            threshold=1.2,
            email="test@example.com"
        )
        mock_serializer.data = {"alert_id": "ALERT-12345"}  # Mock expected response
        mock_serializer_class.return_value = mock_serializer

        data = {
            "base": "USD",
            "target": "EUR",
            "alert_type": "above",
            "threshold": 1.2,
            "email": "test@example.com"
        }
        response = self.client.post(reverse("register-alert"), data, content_type="application/json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(response.data["alert_id"].startswith("ALERT-"))

    @patch("myapp.Serializers.exchangeRateAlertSerializer")
    def test_register_alert_validation_failure(self, mock_serializer_class):
        mock_serializer = MagicMock()
        mock_serializer.is_valid.return_value = False
        mock_serializer.errors = {"email": [ErrorDetail("Enter a valid email address.", code="invalid")]}
        mock_serializer_class.return_value = mock_serializer

        data = {
            "base": "USD",
            "target": "EUR",
            "alert_type": "above",
            "threshold": 1.2,
            "email": "invalid-email"
        }
        response = self.client.post(reverse("register-alert"), data, content_type="application/json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("email", response.data)
        self.assertEqual(response.data["email"], [ErrorDetail("Enter a valid email address.", code="invalid")])

    @patch("myapp.Views.exchangeRateLatestViews.send_alert_email")
    @patch("myapp.Views.exchangeRateLatestViews.requests.get")  # Mocking API call
    def test_alert_email_sent(self, mock_get, mock_send_alert_email):
        # Mock API response from Alpha Vantage with correct structure
        mock_response = {
            "Realtime Currency Exchange Rate": {
                "1. From_Currency Code": "USD",
                "3. To_Currency Code": "EUR",
                "5. Exchange Rate": "1.2",  # Matching the alert threshold
                "6. Last Refreshed": "2025-04-04 12:00:00",
                "8. Bid Price": "1.19",
                "9. Ask Price": "1.21"
            }
        }
        mock_get.return_value.json.return_value = mock_response  # Fake API response
        mock_get.return_value.status_code = 200  # Ensure a successful API call

        # Create an alert in the database (instead of mocking ORM)
        ExchangeRateAlert.objects.create(
            base="USD",
            target="EUR",
            alert_type="above",  # Trigger when rate >= threshold
            threshold=1.2,
            email="test@example.com"
        )

        # Call the API endpoint
        response = self.client.get(reverse("currency-rate", args=["USD", "EUR"]))

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_send_alert_email.assert_called_once_with(
            "test@example.com", "USD", "EUR", 1.2, "above", ANY
        )

    @patch("myapp.Views.exchangeRateLatestViews.send_alert_email")
    @patch("myapp.models.ExchangeRateAlert.objects.filter")
    def test_alert_email_not_sent(self, mock_filter, mock_send_alert_email):
        # Mock database query to return no matching alerts
        mock_filter.return_value.filter.return_value = []

        # Simulate an API call to check for alerts
        response = self.client.get(reverse("currency-rate", args=["USD", "EUR"]))

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_send_alert_email.assert_not_called()  # Ensure no email is sent
