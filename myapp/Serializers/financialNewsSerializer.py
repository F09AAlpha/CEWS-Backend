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
