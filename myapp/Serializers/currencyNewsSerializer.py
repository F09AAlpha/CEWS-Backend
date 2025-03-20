from rest_framework import serializers
from myapp.Models.currencyNewsModel import CurrencyNews


class CurrencyNewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = CurrencyNews
        fields = [
            'id',  # Auto-generated primary key
            'from_currency',  # Source currency code
            'to_currency',  # Target currency code
            'exchange_rate',  # Exchange rate
            'last_refreshed',  # Timestamp of the data
            'created_at',  # Timestamp of when the record was created
        ]
        read_only_fields = ['id', 'created_at']  # Ensure these fields are read-only
