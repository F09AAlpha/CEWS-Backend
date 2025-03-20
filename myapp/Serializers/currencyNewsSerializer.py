from rest_framework import serializers
from myapp.Models.currencyNewsModel import CurrencyNewsAlphaV


class CurrencyNewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = CurrencyNewsAlphaV
        fields = [
            'id',  # Auto-generated primary key
            'title',  # Title of the news article
            'source',  # Source of the news article
            'url',  # URL of the news article
            'summary',  # Summary of the news article
            'sentiment_score',  # Sentiment score
            'sentiment_label',  # Sentiment label
            'publication_date',  # Publication date
            'currency',  # Currency for which the news is relevant
        ]
        read_only_fields = ['id']  # Ensure 'id' is read-only
