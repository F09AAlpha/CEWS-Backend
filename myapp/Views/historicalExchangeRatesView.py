from datetime import datetime
import os
import uuid
import pytz
import requests
from django.db import connection, transaction
from rest_framework.response import Response
from rest_framework.views import APIView
from dotenv import load_dotenv
import psycopg2.extras
import logging

logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()


class FetchHistoricalCurrencyExchangeRates(APIView):

    def post(self, request, from_currency, to_currency, *args, **kwargs):
        API_URL = (
            f"https://www.alphavantage.co/query?function=FX_DAILY"
            f"&from_symbol={from_currency}"
            f"&to_symbol={to_currency}&outputsize=full"
            f"&apikey={os.environ.get('ALPHA_VANTAGE_API_KEY')}"
        )
        try:
            response = requests.get(API_URL)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            return Response({"error": "Failed to fetch data from external API", "details": str(e)}, status=502)

        data = response.json()
        time_series = data.get("Time Series FX (Daily)", {})
        # Generate a table name based on currency pair
        table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

        will_insert = True

        with connection.cursor() as cursor:
            if not time_series:
                logger.error("Error fetching currency rates -- Invalid Currency")
                return Response({"error": "BAD REQUEST -- Currency not found"}, status=400)

            # Check if the table exists
            cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table_name}'
                )
            """)

            table_exists = cursor.fetchone()[0]
            latest_date = None

            if table_exists:
                # Get the latest date from the table
                cursor.execute(f"SELECT MAX(date) FROM {table_name}")
                latest_date = cursor.fetchone()[0]

                if not time_series or (max(time_series.keys()) <= latest_date.isoformat()):
                    will_insert = False

            # Create table if it doesn't exist
            if not table_exists:
                will_insert = True
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        id SERIAL,
                        date DATE NOT NULL PRIMARY KEY UNIQUE,
                        open DECIMAL(10, 5),
                        high DECIMAL(10, 5),
                        low DECIMAL(10, 5),
                        close DECIMAL(10, 5)
                    )
                """)

            # Prepare data for batch insert
            batch_data = [
                (date_str, rate_info.get("1. open"), rate_info.get("2. high"),
                 rate_info.get("3. low"), rate_info.get("4. close"))
                for date_str, rate_info in time_series.items()
                if latest_date is None or date_str > latest_date.isoformat()
            ]

            status_code = 200

            # Batch insert data
            if will_insert:
                status_code = 201
                with transaction.atomic():
                    psycopg2.extras.execute_values(
                        cursor,
                        f"""
                        INSERT INTO {table_name} (date, open, high, low, close)
                        VALUES %s
                        ON CONFLICT (date) DO NOTHING
                        """,
                        batch_data
                    )
            return Response({
                "data_source": "Alpha Vantage",
                "dataset_type": "historical_currency_exchange_rate",
                "dataset_id": f"currency-{from_currency}-{to_currency}-{datetime.now(pytz.UTC).strftime('%Y')}-2014",
                "time_object": {"timestamp": datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
                                "timezone": "UTC"},
                "event": [{
                    "event_type": "historical_currency_rates",
                    "event_id": f"CE-{datetime.now(pytz.UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
                    "attributes": {
                        "base": data["Meta Data"]["2. From Symbol"],
                        "target": data["Meta Data"]["3. To Symbol"],
                        "data": data["Time Series FX (Daily)"]
                        }
                    }],
                },
                status=status_code
            )
