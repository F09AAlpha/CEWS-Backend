from rest_framework import serializers
from myapp.Models.exchangeRateLatestModel import CurrencyEvent


class CurrencyRateSerializer(serializers.ModelSerializer):
    """Serializer for currency rate responses"""
    class Meta:
        model = CurrencyEvent
        fields = ['base', 'target', 'rate', 'timestamp', 'source', 'event_id']
