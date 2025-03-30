from rest_framework import serializers


class EventTimeObjectSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    duration = serializers.IntegerField()
    duration_unit = serializers.CharField()
    timezone = serializers.CharField()


class EventAttributesSerializer(serializers.Serializer):
    base_currency = serializers.CharField(max_length=10)
    target_currency = serializers.CharField(max_length=10)
    current_volatility = serializers.FloatField()
    average_volatility = serializers.FloatField()
    volatility_level = serializers.CharField()
    analysis_period_days = serializers.IntegerField()
    trend = serializers.CharField()
    data_points = serializers.IntegerField()
    confidence_score = serializers.FloatField(required=False)


class EventSerializer(serializers.Serializer):
    time_object = EventTimeObjectSerializer()
    event_type = serializers.CharField()
    attributes = EventAttributesSerializer()


class DatasetTimeObjectSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    timezone = serializers.CharField()


class VolatilityAnalysisSerializer(serializers.Serializer):
    """
    ADAGE 3.0 compliant serializer for volatility analysis
    """
    data_source = serializers.CharField()
    dataset_type = serializers.CharField()
    dataset_id = serializers.CharField()
    time_object = DatasetTimeObjectSerializer()
    events = EventSerializer(many=True)
