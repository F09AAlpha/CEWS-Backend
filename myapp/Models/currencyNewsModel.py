from django.db import models


class CurrencyNewsAlphaV(models.Model):
    title = models.CharField(max_length=255)  # Title of the news article
    source = models.CharField(max_length=100)  # Source of the news article
    url = models.URLField(unique=True)  # URL of the news article (unique)
    summary = models.TextField()  # Summary of the news article
    sentiment_score = models.FloatField()  # Sentiment score (e.g., -1 to 1)
    sentiment_label = models.CharField(max_length=20)  # Sentiment label (e.g., "Bullish")
    publication_date = models.DateTimeField()  # Publication date of the article
    currency = models.CharField(max_length=10)  # Currency for which the news is relevant

    class Meta:
        ordering = ['-publication_date']  # Order by publication date (newest first)

    def __str__(self):
        return f"{self.title} - {self.source}"
