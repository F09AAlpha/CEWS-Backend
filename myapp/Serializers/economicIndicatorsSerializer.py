from rest_framework import serializers


class TimeObjectSerializer(serializers.Serializer):
    """Serializer for time object information in ADAGE 3.0 format."""
    timestamp = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S.%f", help_text="Timestamp rep")
    duration = serializers.IntegerField(required=False, default=365, help_text="Event duration")
    duration_unit = serializers.CharField(required=False, default="days", help_text="Unit of duration measurement")
    timezone = serializers.CharField(default="UTC", help_text="Timezone of the timestamp")


class EconomicIndicatorEventSerializer(serializers.Serializer):
    """Serializer for Economic Indicator events in ADAGE 3.0 format."""
    time_object = TimeObjectSerializer(help_text="Time information for the event")
    event_type = serializers.CharField(default="economic_indicator", help_text="Type of event")
    event_id = serializers.CharField(help_text="Unique event identifier")
    attributes = serializers.DictField(help_text="Event data attributes")


class EconomicIndicatorsResponseSerializer(serializers.Serializer):
    """
    Main serializer for Economic Indicator data in ADAGE 3.0 format.
    This represents the top-level structure of the response.
    """
    data_source = serializers.CharField(default="Alpha Vantage", help_text="Origin of the data")
    dataset_type = serializers.CharField(default="annual_economic_indicators", help_text="Classification of the dataset")
    dataset_id = serializers.CharField(help_text="Unique identifier for the dataset")
    time_object = TimeObjectSerializer(help_text="Time information for the dataset")
    events = EconomicIndicatorEventSerializer(many=True, help_text="List of events in the dataset")
