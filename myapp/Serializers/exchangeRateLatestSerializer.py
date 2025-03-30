from rest_framework import serializers


class TimeObjectSerializer(serializers.Serializer):
    """Serializer for time object information in ADAGE 3.0 format."""
    timestamp = serializers.CharField(help_text="Timestamp representation")
    timezone = serializers.CharField(help_text="Timezone of the timestamp")
    duration = serializers.IntegerField(required=False, help_text="Event duration")
    duration_unit = serializers.CharField(required=False, help_text="Unit of duration measurement")


class CurrencyRateAttributesSerializer(serializers.Serializer):
    """Serializer for currency rate event attributes in ADAGE 3.0 format."""
    base = serializers.CharField(max_length=3, help_text="Base currency code")
    target = serializers.CharField(max_length=3, help_text="Target currency code")
    rate = serializers.FloatField(help_text="Exchange rate value")
    bid_price = serializers.CharField(required=False, allow_null=True, help_text="Bid price if available")
    ask_price = serializers.CharField(required=False, allow_null=True, help_text="Ask price if available")
    source = serializers.CharField(help_text="Data source name")


class CurrencyRateEventSerializer(serializers.Serializer):
    """Serializer for currency rate events in ADAGE 3.0 format."""
    time_object = TimeObjectSerializer(help_text="Time information for the event")
    event_type = serializers.CharField(help_text="Type of event")
    event_id = serializers.CharField(help_text="Unique event identifier")
    attributes = CurrencyRateAttributesSerializer(help_text="Event data attributes")


class AdageCurrencyRateSerializer(serializers.Serializer):
    """
    Main serializer for currency exchange rate data in ADAGE 3.0 format.
    This represents the top-level structure of the response.
    """
    data_source = serializers.CharField(help_text="Origin of the data")
    dataset_type = serializers.CharField(help_text="Classification of the dataset")
    dataset_id = serializers.CharField(help_text="Unique identifier for the dataset")
    time_object = TimeObjectSerializer(help_text="Time information for the dataset")
    events = CurrencyRateEventSerializer(many=True, help_text="List of events in the dataset")
