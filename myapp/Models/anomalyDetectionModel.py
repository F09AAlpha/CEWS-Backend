from django.db import models
from django.core.validators import MinLengthValidator, RegexValidator
from django.utils.translation import gettext_lazy as _


class AnomalyDetectionResult(models.Model):
    """Model to store anomaly detection results."""

    # Currency pair information
    base_currency = models.CharField(
        max_length=3,
        validators=[
            MinLengthValidator(3),
            RegexValidator(r'^[A-Z]{3}$', 'Currency code must be 3 uppercase letters')
        ],
        help_text=_("Base currency code (e.g., USD)")
    )
    target_currency = models.CharField(
        max_length=3,
        validators=[
            MinLengthValidator(3),
            RegexValidator(r'^[A-Z]{3}$', 'Currency code must be 3 uppercase letters')
        ],
        help_text=_("Target currency code (e.g., EUR)")
    )

    # Analysis metadata
    analysis_date = models.DateTimeField(
        auto_now_add=True,
        help_text=_("When the analysis was performed")
    )
    analysis_period_days = models.PositiveIntegerField(
        default=30,
        help_text=_("Analysis period in days")
    )
    anomaly_count = models.PositiveIntegerField(
        help_text=_("Number of anomalies detected")
    )

    # Result data
    result_data = models.JSONField(
        help_text=_("Complete anomaly detection results in ADAGE 3.0 format")
    )

    class Meta:
        verbose_name = _("Anomaly Detection Result")
        verbose_name_plural = _("Anomaly Detection Results")
        indexes = [
            models.Index(fields=['base_currency', 'target_currency']),
            models.Index(fields=['analysis_date']),
        ]
        ordering = ['-analysis_date']

    def __str__(self):
        date_str = self.analysis_date.strftime('%Y-%m-%d %H:%M')
        return f"Anomaly: {self.base_currency}/{self.target_currency} ({date_str})"


class AnomalyPoint(models.Model):
    """Model to store individual anomaly points detected."""

    result = models.ForeignKey(
        AnomalyDetectionResult,
        on_delete=models.CASCADE,
        related_name='anomaly_points',
        help_text=_("Related anomaly detection result")
    )

    timestamp = models.DateTimeField(
        help_text=_("When the anomaly occurred")
    )

    rate = models.FloatField(
        help_text=_("Exchange rate value")
    )

    z_score = models.FloatField(
        help_text=_("Z-score of the anomaly")
    )

    percent_change = models.FloatField(
        help_text=_("Percent change from previous value")
    )

    class Meta:
        verbose_name = _("Anomaly Point")
        verbose_name_plural = _("Anomaly Points")
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"Anomaly at {self.timestamp.strftime('%Y-%m-%d')}: {self.z_score:.2f}"
