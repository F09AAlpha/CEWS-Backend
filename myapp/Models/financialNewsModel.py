from django.db import models


class FinancialNews(models.Model):
    symbol = models.CharField(max_length=10)  # Stock symbol (e.g., "AAPL")
    date = models.DateTimeField()  # Date of the financial data
    open_price = models.FloatField(null = True)  # Opening price
    high_price = models.FloatField(null = True)  # Highest price of the day
    low_price = models.FloatField(null = True)  # Lowest price of the day
    close_price = models.FloatField(null = True)  # Closing price
    volume = models.IntegerField()  # Trading volume

    class Meta:
        unique_together = ('symbol', 'date')  # Prevent duplicate entries

    def __str__(self):
        return f"{self.symbol} - {self.date}"
