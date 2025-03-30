from django.db import models


class ExchangeRateAlert(models.Model):
    alert_id = models.CharField(max_length=100, unique=True, blank=True, null=True)  # Add the alert_id field
    base = models.CharField(max_length=3)
    target = models.CharField(max_length=3)
    alert_type = models.CharField(max_length=10, choices=[('above', 'Above'), ('below', 'Below')])
    threshold = models.DecimalField(max_digits=10, decimal_places=6)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Alert for {self.base}/{self.target} at {self.threshold} ({self.alert_type})"
