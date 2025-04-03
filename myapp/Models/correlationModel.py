from django.db import models


class CorrelationResult(models.Model):
    """
    Model for storing the results of correlation analyses between currency exchange rates
    and various factors such as economic indicators and news sentiment.
    """
    base_currency = models.CharField(max_length=3)
    target_currency = models.CharField(max_length=3)
    analysis_date = models.DateTimeField(auto_now_add=True)
    lookback_days = models.IntegerField(default=90)

    # Store correlation values as JSON
    exchange_news_correlation = models.JSONField(default=dict)
    exchange_economic_correlation = models.JSONField(default=dict)
    volatility_news_correlation = models.JSONField(default=dict)
    volatility_economic_correlation = models.JSONField(default=dict)

    # Top factors with coefficient values
    top_influencing_factors = models.JSONField(default=dict)

    # Correlation confidence metrics
    confidence_score = models.FloatField()
    data_completeness = models.FloatField()  # Percentage of days with complete data

    class Meta:
        ordering = ["-analysis_date"]
        indexes = [
            models.Index(fields=["base_currency", "target_currency"], name="correlation_currency_pair_idx"),
            models.Index(fields=["analysis_date"], name="correlation_date_idx"),
        ]
        unique_together = [["base_currency", "target_currency", "analysis_date"]]

    def __str__(self):
        return f"Correlation Analysis {self.base_currency}/{self.target_currency} ({self.analysis_date})"
