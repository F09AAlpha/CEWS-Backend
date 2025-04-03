from django.core.management.base import BaseCommand
import requests
import os
import logging
from django.utils.dateparse import parse_datetime
from datetime import timezone
from myapp.Models.currencyNewsModel import CurrencyNewsAlphaV
from myapp.Serializers.currencyNewsSerializer import CurrencyNewsSerializer

logger = logging.getLogger(__name__)

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class Command(BaseCommand):
    help = 'Fetch and store news for major currencies'

    def add_arguments(self, parser):
        parser.add_argument(
            '--currencies',
            nargs='+',
            type=str,
            default=['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF'],
            help='List of currency codes to fetch news for'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=10,
            help='Number of news items to fetch per currency'
        )

    def handle(self, *args, **options):
        currencies = options['currencies']
        limit = options['limit']

        for currency in currencies:
            self.stdout.write(f"Fetching news for {currency}...")
            self.fetch_currency_news(currency, limit)

        self.stdout.write(self.style.SUCCESS('Currency news fetching completed'))

    def fetch_currency_news(self, currency, limit):
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": f"{currency},forex",
            "apikey": ALPHA_VANTAGE_API_KEY,
            "sort": "LATEST",
            "limit": limit
        }

        try:
            response = requests.get(ALPHA_VANTAGE_URL, params=params)
            response.raise_for_status()

            data = response.json()
            news_data = data.get("feed", [])

            if news_data:
                stored_count = 0
                for article in news_data:
                    publication_date = parse_datetime(article.get("time_published"))
                    aware_publication_date = publication_date.replace(tzinfo=timezone.utc)

                    if not CurrencyNewsAlphaV.objects.filter(url=article.get("url")).exists():
                        news_data = {
                            "title": article.get("title"),
                            "source": article.get("source"),
                            "url": article.get("url"),
                            "summary": article.get("summary"),
                            "sentiment_score": float(article.get("overall_sentiment_score", 0)),
                            "sentiment_label": article.get("overall_sentiment_label", "neutral"),
                            "publication_date": aware_publication_date,
                            "currency": currency
                        }

                        serializer = CurrencyNewsSerializer(data=news_data)
                        if serializer.is_valid():
                            serializer.save()
                            stored_count += 1

                self.stdout.write(f"Stored {stored_count} new articles for {currency}")
            else:
                self.stdout.write(f"No news found for {currency}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error fetching news for {currency}: {str(e)}"))
