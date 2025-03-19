from django.db import models


class CurrencyNews(models.Model):
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=100)
    url = models.URLField(unique=True)
    published_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
