from rest_framework import serializers
from myapp.Models.financialNewsModel import FinancialNews


class FinancialNewsSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = FinancialNews
        fields = ["id", "title", "source", "url", "published_at"]
