from rest_framework import serializers


class CorrelationFactorSerializer(serializers.Serializer):
    """
    Serializer for correlation factors in ADAGE 3.0 format
    """
    factor_name = serializers.CharField()
    impact_level = serializers.CharField()  # high, medium, low
    correlation_coefficient = serializers.FloatField()
    factor_type = serializers.CharField()  # news, economic


class CorrelationValueSerializer(serializers.Serializer):
    """
    Serializer for correlation values in ADAGE 3.0 format
    """
    news_sentiment = serializers.DictField(required=False)
    economic_indicators = serializers.DictField(required=False)
    volatility_news = serializers.DictField(required=False)


class CorrelationEventTimeObjectSerializer(serializers.Serializer):
    """
    Serializer for correlation event time object in ADAGE 3.0 format
    """
    timestamp = serializers.DateTimeField()
    duration = serializers.IntegerField()
    duration_unit = serializers.CharField()
    timezone = serializers.CharField()


class CorrelationAttributesSerializer(serializers.Serializer):
    """
    Serializer for correlation attributes in ADAGE 3.0 format
    """
    base_currency = serializers.CharField(max_length=3)
    target_currency = serializers.CharField(max_length=3)
    confidence_score = serializers.FloatField()
    data_completeness = serializers.FloatField()
    analysis_period_days = serializers.IntegerField()
    influencing_factors = CorrelationFactorSerializer(many=True)
    correlations = CorrelationValueSerializer()


class CorrelationEventSerializer(serializers.Serializer):
    """
    Serializer for correlation event in ADAGE 3.0 format
    """
    time_object = CorrelationEventTimeObjectSerializer()
    event_type = serializers.CharField()
    attributes = CorrelationAttributesSerializer()


class DatasetTimeObjectSerializer(serializers.Serializer):
    """
    Serializer for dataset time object in ADAGE 3.0 format
    """
    timestamp = serializers.DateTimeField()
    timezone = serializers.CharField()


class CorrelationAnalysisSerializer(serializers.Serializer):
    """
    ADAGE 3.0 compliant serializer for correlation analysis
    """
    data_source = serializers.CharField()
    dataset_type = serializers.CharField()
    dataset_id = serializers.CharField()
    time_object = DatasetTimeObjectSerializer()
    events = CorrelationEventSerializer(many=True)
