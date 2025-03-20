from django.db import models


class FinancialNews(models.Model):
    symbol = models.CharField(max_length=10)  # Stock symbol (e.g., "AAPL")
    date = models.DateTimeField()  # Date of the financial data
    open_price = models.FloatField()  # Opening price
    high_price = models.FloatField()  # Highest price of the day
    low_price = models.FloatField()  # Lowest price of the day
    close_price = models.FloatField()  # Closing price
    volume = models.IntegerField()  # Trading volume

    class Meta:
        unique_together = ('symbol', 'date')  # Ensure no duplicate entries for the same symbol and date

    def __str__(self):
        return f"{self.symbol} - {self.date}"
