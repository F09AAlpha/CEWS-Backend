from django.db import models
import uuid


class CurrencyEvent(models.Model):
    """Model for storing currency exchange rate events"""
    event_id = models.CharField(max_length=50, primary_key=True)
    base = models.CharField(max_length=10)
    target = models.CharField(max_length=10)
    rate = models.DecimalField(max_digits=20, decimal_places=10)
    timestamp = models.DateTimeField()
    source = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['base', 'target']),
            models.Index(fields=['timestamp']),
        ]

    @staticmethod
    def generate_event_id(base, target):
        """Generate a unique event ID for currency events"""
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        unique_id = uuid.uuid4().hex[:8]
        return f"CM-{timestamp}-{base}-{target}-{unique_id}"
