import unittest
from django.test import TestCase

from myapp.Serializers.exchangeRateLatestSerializer import (
    TimeObjectSerializer,
    CurrencyRateAttributesSerializer,
    AdageCurrencyRateSerializer
)
from myapp.Views.exchangeRateLatestViews import is_valid_email


class ExchangeRateSerializerTests(TestCase):
    """Unit tests for the Exchange Rate Serializers"""

    def test_time_object_serializer_valid_data(self):
        """Test TimeObjectSerializer with valid data"""
        data = {
            "timestamp": "2023-04-01 12:00:00.000000",
            "timezone": "UTC"
        }
        serializer = TimeObjectSerializer(data=data)
        self.assertTrue(serializer.is_valid())

    def test_time_object_serializer_invalid_data(self):
        """Test TimeObjectSerializer with missing required fields"""
        data = {"timestamp": "2023-04-01 12:00:00.000000"}  # Missing timezone
        serializer = TimeObjectSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('timezone', serializer.errors)

    def test_currency_rate_attributes_serializer_valid_data(self):
        """Test CurrencyRateAttributesSerializer with valid data"""
        data = {
            "base": "USD",
            "target": "EUR",
            "rate": 0.92,
            "source": "Alpha Vantage"
        }
        serializer = CurrencyRateAttributesSerializer(data=data)
        self.assertTrue(serializer.is_valid())

    def test_currency_rate_attributes_serializer_invalid_data(self):
        """Test CurrencyRateAttributesSerializer with invalid data"""
        data = {
            "base": "USDD",  # Too long
            "target": "EUR",
            "rate": 0.92,
            "source": "Alpha Vantage"
        }
        serializer = CurrencyRateAttributesSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('base', serializer.errors)

    def test_complete_adage_serialization(self):
        """Test full ADAGE format serialization"""
        data = {
            "data_source": "Alpha Vantage",
            "dataset_type": "currency_exchange_rate",
            "dataset_id": "currency-USD-EUR-20230401",
            "time_object": {
                "timestamp": "2023-04-01 12:00:00.000000",
                "timezone": "UTC"
            },
            "events": [
                {
                    "time_object": {
                        "timestamp": "2023-04-01 12:00:00.000000",
                        "timezone": "UTC",
                        "duration": 0,
                        "duration_unit": "second"
                    },
                    "event_type": "currency_rate",
                    "event_id": "CE-20230401-12345678",
                    "attributes": {
                        "base": "USD",
                        "target": "EUR",
                        "rate": 0.92,
                        "source": "Alpha Vantage"
                    }
                }
            ]
        }
        serializer = AdageCurrencyRateSerializer(data=data)
        self.assertTrue(serializer.is_valid())


class ExchangeRateViewHelperTests(TestCase):
    """Unit tests for helper functions in ExchangeRateView"""

    def test_valid_email(self):
        """Test email validation with valid emails"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+label@domain.org"
        ]
        for email in valid_emails:
            self.assertTrue(is_valid_email(email))

    def test_invalid_email(self):
        """Test email validation with invalid emails"""
        invalid_emails = [
            "test@",
            "user@domain",
            "@domain.com",
            "user name@domain.com"
        ]
        for email in invalid_emails:
            self.assertFalse(is_valid_email(email))


if __name__ == '__main__':
    unittest.main()
