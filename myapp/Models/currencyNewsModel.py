from django.db import models


class CurrencyNews(models.Model):
    from_currency = models.CharField(max_length=10)  # Currency code (e.g., "USD")
    to_currency = models.CharField(max_length=10)  # Currency code (e.g., "EUR")
    exchange_rate = models.FloatField()  # Exchange rate
    last_refreshed = models.DateTimeField()  # Timestamp of the data
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp of when the record was created

    class Meta:
        unique_together = ('from_currency', 'to_currency', 'last_refreshed')  # Prevent duplicate entries

    def __str__(self):
        return f"{self.from_currency} to {self.to_currency} - {self.exchange_rate}"
