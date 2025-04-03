from rest_framework import serializers
from django.utils import timezone


class AnomalyPointSerializer(serializers.Serializer):
    """Serializer for anomaly data points in the API response."""

    timestamp = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S.%fZ")
    rate = serializers.FloatField()
    z_score = serializers.FloatField()
    percent_change = serializers.FloatField()


class AnomalyDetectionRequestSerializer(serializers.Serializer):
    """Serializer for anomaly detection request parameters."""

    base = serializers.CharField(required=True, max_length=3, min_length=3)
    target = serializers.CharField(required=True, max_length=3, min_length=3)
    days = serializers.IntegerField(required=False, default=30, min_value=1, max_value=365)

    def validate_base(self, value):
        """Validate base currency code."""
        value = value.upper()
        if not value.isalpha():
            raise serializers.ValidationError("Base currency code must be 3 letters.")
        return value

    def validate_target(self, value):
        """Validate target currency code."""
        value = value.upper()
        if not value.isalpha():
            raise serializers.ValidationError("Target currency code must be 3 letters.")
        return value

    def validate(self, data):
        """Validate that base and target currencies are different."""
        if data['base'] == data['target']:
            raise serializers.ValidationError("Base and target currencies must be different.")
        return data


# ADAGE 3.0 Serializers
class EventTimeObjectSerializer(serializers.Serializer):
    """Serializer for the Event Time Object in ADAGE 3.0 format."""

    timestamp = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S.%fZ")
    duration = serializers.IntegerField(default=0)
    duration_unit = serializers.CharField(default="day")
    timezone = serializers.CharField(default="GMT+11")


class EventAttributeSerializer(serializers.Serializer):
    """Serializer for Event Attributes in ADAGE 3.0 format."""

    base_currency = serializers.CharField()
    target_currency = serializers.CharField()
    rate = serializers.FloatField()
    z_score = serializers.FloatField()
    percent_change = serializers.FloatField()


class EventSerializer(serializers.Serializer):
    """Serializer for Events in ADAGE 3.0 format."""

    time_object = EventTimeObjectSerializer()
    event_type = serializers.CharField(default="exchange_rate_anomaly")
    attribute = EventAttributeSerializer()


class DatasetTimeObjectSerializer(serializers.Serializer):
    """Serializer for Dataset Time Object in ADAGE 3.0 format."""

    timestamp = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S.%fZ")
    timezone = serializers.CharField(default="GMT+11")


class ADAGE30ResponseSerializer(serializers.Serializer):
    """Serializer for ADAGE 3.0 compliant response."""

    data_source = serializers.CharField(default="Alpha Vantage")
    dataset_type = serializers.CharField(default="Currency Exchange Rates")
    dataset_id = serializers.CharField()
    time_object = DatasetTimeObjectSerializer()
    events = EventSerializer(many=True)


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses."""

    detail = serializers.CharField()


class ADAGEConverter:
    """Utility class to convert standard response to ADAGE 3.0 format."""

    @staticmethod
    def convert_to_adage_format(standard_response):
        """
        Convert standard anomaly detection response to ADAGE 3.0 format.

        Args:
            standard_response (dict): Standard API response

        Returns:
            dict: ADAGE 3.0 formatted response
        """
        # Extract base data
        base_currency = standard_response.get('base')
        target_currency = standard_response.get('target')

        # Create events array
        events = []
        for anomaly in standard_response.get('anomaly_points', []):
            event = {
                'time_object': {
                    'timestamp': anomaly.get('timestamp'),
                    'duration': 0,
                    'duration_unit': 'day',
                    'timezone': 'GMT+11'
                },
                'event_type': 'exchange_rate_anomaly',
                'attribute': {
                    'base_currency': base_currency,
                    'target_currency': target_currency,
                    'rate': anomaly.get('rate'),
                    'z_score': anomaly.get('z_score'),
                    'percent_change': anomaly.get('percent_change')
                }
            }
            events.append(event)

        # Create ADAGE 3.0 response
        adage_response = {
            'data_source': 'Alpha Vantage',
            'dataset_type': 'Currency Exchange Rates',
            'dataset_id': f"exchange_anomaly_{base_currency}_{target_currency}",
            'time_object': {
                'timestamp': timezone.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                'timezone': 'GMT+11'
            },
            'events': events
        }

        return adage_response
