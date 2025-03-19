from rest_framework import serializers
from myapp.Models.currencyNewsModel import CurrencyNews


class CurrencyNewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = CurrencyNews
        fields = ['id', 'title', 'source', 'url', 'published_at']
