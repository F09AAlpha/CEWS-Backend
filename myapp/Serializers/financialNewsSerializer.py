from rest_framework import serializers
from myapp.Models.financialNewsModel import FinancialNewsAlphaV


class FinancialNewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = FinancialNewsAlphaV
        fields = [
            'id',  # Auto-generated primary key
            'title',  # Title of the news article
            'source',  # Source of the news article
            'url',  # URL of the news article
            'summary',  # Summary of the news article
            'sentiment_score',  # Sentiment score
            'sentiment_label',  # Sentiment label
            'publication_date',  # Publication date
            'symbol',  # Stock symbol for which the news is relevant
        ]
        read_only_fields = ['id']  # Ensure 'id' is read-only


# ADAGE 3.0 Data Model Serializers
class TimeObjectSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    duration = serializers.IntegerField(default=0)
    duration_unit = serializers.CharField(default="second")
    timezone = serializers.CharField(default="UTC")


class FinancialNewsAttributesSerializer(serializers.Serializer):
    title = serializers.CharField()
    source = serializers.CharField()
    url = serializers.URLField()
    summary = serializers.CharField()
    sentiment_score = serializers.FloatField()
    sentiment_label = serializers.CharField()
    symbol = serializers.CharField()


class FinancialNewsEventSerializer(serializers.Serializer):
    time_object = TimeObjectSerializer()
    event_type = serializers.CharField(default="financial_news")
    event_id = serializers.CharField()
    attributes = FinancialNewsAttributesSerializer()


class AdageFinancialNewsDatasetSerializer(serializers.Serializer):
    data_source = serializers.CharField(default="Alpha Vantage")
    dataset_type = serializers.CharField(default="financial_news")
    dataset_id = serializers.CharField()
    time_object = TimeObjectSerializer()
    events = FinancialNewsEventSerializer(many=True)
