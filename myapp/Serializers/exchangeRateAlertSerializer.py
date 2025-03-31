from rest_framework import serializers
from myapp.Models.exchangeRateAlertModel import ExchangeRateAlert


class ExchangeRateAlertSerializer(serializers.ModelSerializer):
    # Include the alert_id field in the serializer but not in the model
    alert_id = serializers.CharField(required=False, allow_null=True, read_only=True)  # We'll allow it to be read-only

    class Meta:
        model = ExchangeRateAlert
        fields = ['alert_id', 'base', 'target', 'alert_type', 'threshold', 'email', 'created_at']
