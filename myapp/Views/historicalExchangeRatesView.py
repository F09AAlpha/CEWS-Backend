import os
import requests
from django.db import connection, transaction
from rest_framework.response import Response
from rest_framework.views import APIView
from dotenv import load_dotenv
import psycopg2.extras

# Load environment variables from the .env file
load_dotenv()

class FetchHistoricalCurrencyExchangeRates(APIView):

    def get(self, request, from_currency, to_currency, *args, **kwargs):
        API_URL = (
            f"https://www.alphavantage.co/query?function=FX_DAILY"
            f"&from_symbol={from_currency}"
            f"&to_symbol={to_currency}&outputsize=full&"
            f"&apikey={os.environ.get('ALPHA_VANTAGE_API_KEY')}"
        )

        try:
            response = requests.get(API_URL)
        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=500)

        data = response.json()
        time_series = data.get("Time Series FX (Daily)", {})

        # Generate a table name based on currency pair
        table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

        # Check if the table exists
        with connection.cursor() as cursor:
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
                if not (time_series.keys()) or (max(time_series.keys())) <= latest_date.isoformat():
                    return Response({"Message": "Data fetched but no new data available",
                                    "Result": data},
                                    status=200)

            # Create table if it doesn't exist
            if not table_exists:
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        date DATE NOT NULL UNIQUE,
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

        return Response({"Message": "Data fetched and stored successfully", "Result": data}, status=201)
'''
            count = 0
            
            # Insert data into the table
            for date_str, rate_info in time_series.items():
                date = date_str  # Convert to a date object if necessary
                if latest_date is None or date > latest_date.isoformat():
                    open_rate = rate_info.get("1. open")
                    high_rate = rate_info.get("2. high")
                    low_rate = rate_info.get("3. low")
                    close_rate = rate_info.get("4. close")

                    cursor.execute(f"""
                        INSERT INTO {table_name} (date, open, high, low, close)
                        VALUES (%s, %s, %s, %s, %s)
                    """, [date, open_rate, high_rate, low_rate, close_rate])
                    
                    count = count + 1
                    print(count)
                
                if count == 1500:
                    return Response({"Message": "Past 5 years data fetched and stored successfully", "Result": data}, status=201)

        return Response({"Message": "Data fetched and stored successfully", "Result": data}, status=201)'''
