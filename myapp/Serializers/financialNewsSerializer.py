from rest_framework import serializers
from myapp.Models.financialNewsModel import FinancialNews


class FinancialNewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = FinancialNews
        fields = [
            'id',  # Auto-generated primary key
            'symbol',  # Stock symbol
            'date',  # Date of the financial data
            'open_price',  # Opening price
            'high_price',  # Highest price of the day
            'low_price',  # Lowest price of the day
            'close_price',  # Closing price
            'volume',  # Trading volume
        ]
        read_only_fields = ['id']  # Ensure 'id' is read-only
