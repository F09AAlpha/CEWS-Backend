from rest_framework import serializers
from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert


class ExchangeRateAlertSerializer(serializers.ModelSerializer):
    alert_id = serializers.CharField(required=False, allow_null=True) 

    class Meta:
        model = ExchangeRateAlert
        fields = ['alert_id', 'base', 'target', 'alert_type', 'threshold', 'email', 'created_at']
