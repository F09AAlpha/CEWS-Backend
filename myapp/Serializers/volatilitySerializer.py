from rest_framework import serializers


class VolatilityAnalysisSerializer(serializers.Serializer):
    base = serializers.CharField(max_length=3)
    target = serializers.CharField(max_length=3)
    current_volatility = serializers.FloatField()
    average_volatility = serializers.FloatField()
    volatility_level = serializers.CharField()
    analysis_period_days = serializers.IntegerField()
    trend = serializers.CharField()
