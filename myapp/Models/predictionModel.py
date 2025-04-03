from django.db import models
from django.utils.translation import gettext_lazy as _


class CurrencyPrediction(models.Model):
    """
    Model for storing currency exchange rate predictions
    generated using the Lag-Llama model or other forecasting methods.
    """
    base_currency = models.CharField(max_length=3)
    target_currency = models.CharField(max_length=3)
    prediction_date = models.DateTimeField(auto_now_add=True)
    forecast_horizon = models.IntegerField(help_text="Number of days ahead for the prediction")

    # Current rate and change
    current_rate = models.FloatField(default=0.0, help_text="Current exchange rate at prediction time")
    change_percent = models.FloatField(default=0.0, help_text="Predicted percentage change from current rate")

    # Store prediction values as JSON
    mean_predictions = models.JSONField(default=dict, help_text="Mean predictions for each forecasted day")
    lower_bound = models.JSONField(default=dict, help_text="Lower bound (10th percentile) for each forecasted day")
    upper_bound = models.JSONField(default=dict, help_text="Upper bound (90th percentile) for each forecasted day")

    # Metadata and additional info
    model_version = models.CharField(max_length=50, help_text="Version of the model used")
    confidence_score = models.FloatField(help_text="Confidence score of the prediction")
    input_data_range = models.CharField(max_length=100, help_text="Range of dates used for input data")

    # Factors used for prediction
    used_correlation_data = models.BooleanField(default=False, help_text="Whether correlation data was used")
    used_news_sentiment = models.BooleanField(default=False, help_text="Whether news sentiment was used")
    used_economic_indicators = models.BooleanField(default=False, help_text="Whether economic indicators were used")

    # Add a new field to track if anomaly detection was used in the prediction
    used_anomaly_detection = models.BooleanField(
        default=False,
        help_text=_("Whether anomaly detection was used in the prediction")
    )

    class Meta:
        ordering = ["-prediction_date"]
        indexes = [
            models.Index(fields=["base_currency", "target_currency"], name="prediction_currency_pair_idx"),
            models.Index(fields=["prediction_date"], name="prediction_date_idx"),
        ]
        unique_together = [["base_currency", "target_currency", "prediction_date", "forecast_horizon"]]

    def __str__(self):
        return (f"Prediction for {self.base_currency}/{self.target_currency} "
                f"({self.prediction_date}) - {self.forecast_horizon} days")
