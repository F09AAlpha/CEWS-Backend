import os
import requests
from django.db import connection, transaction
from rest_framework.response import Response
from rest_framework.views import APIView
from dotenv import load_dotenv
import psycopg2.extras
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

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
            return Response({"error": str(e)}, status=500)

        data = response.json()
        time_series = data.get("Time Series FX (Daily)", {})
        # Generate a table name based on currency pair
        table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

        will_insert = True

        with connection.cursor() as cursor:
            if not time_series:
                cursor.execute(f"""
                    SELECT date, open, high, low, close FROM {table_name} ORDER BY date
                """)
                rows = cursor.fetchall()
                data = [
                    {
                        "date": row[0].isoformat(),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4])
                    }
                    for row in rows
                ]

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

            # Batch insert data
            if will_insert:
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

        if will_insert:
            return Response({"Message": "Data fetched and stored successfully", "Result": data}, status=201)
        return Response({"Message": "Data fetched but no new data available",
                         "From": from_currency,
                         "To": to_currency,
                         "Data": data},
                        status=201
                        )
