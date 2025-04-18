from django.db import models


class AnnualEconomicIndicator(models.Model):
    # Model for storing Annual Economic Indicators like GDP and Inflation

    id = models.AutoField(primary_key=True)
    date = models.DateField(unique=True)
    real_gdp = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    inflation = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)

    class Meta:
        ordering = ["-date"]
        indexes = [
            models.Index(fields=["date"], name="annual_indicator_date_idx"),
        ]

    def __str__(self):
        return f"Annual Indicators ({self.date})"


class MonthlyEconomicIndicator(models.Model):
    # Model for storing Monthly Economic Indicators like CPI, Unemployment rate, etc.

    id = models.AutoField(primary_key=True)
    date = models.DateField(unique=True)
    cpi = models.DecimalField(max_digits=6, decimal_places=3, null=True, blank=True)
    unemployment_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    federal_funds_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    treasury_yield = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    class Meta:
        ordering = ["-date"]
        indexes = [
            models.Index(fields=["date"], name="monthly_indicator_date_idx"),
        ]

    def __str__(self):
        return f"Monthly Indicators ({self.date})"
