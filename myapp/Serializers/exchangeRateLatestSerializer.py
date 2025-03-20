from rest_framework import serializers


class CurrencyRateSerializer(serializers.Serializer):
    """
    Serializer for currency exchange rate data.
    This matches the CurrencyRate schema from the OpenAPI specification.
    """
    base = serializers.CharField(max_length=3, help_text="Base currency code")
    target = serializers.CharField(max_length=3, help_text="Target currency code")
    rate = serializers.CharField(help_text="Exchange rate")
    timestamp = serializers.CharField(help_text="Last refreshed timestamp")
    source = serializers.CharField(help_text="Data source")
