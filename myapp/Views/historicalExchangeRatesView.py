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

def saveGraph(from_currency, to_currency, table_name):
    with connection.cursor() as cursor:
        one_week_ago = datetime.now() - timedelta(days=8)
        one_month_ago = datetime.now() - timedelta(days=30)
        six_months_ago = datetime.now() - timedelta(days=181)
        one_year_ago = datetime.now() - timedelta(days=366)
        five_years_ago = datetime.now() - timedelta(days=(366*5))
        
        graph_list = [one_week_ago, one_month_ago, six_months_ago, one_year_ago, five_years_ago]
        
        for graph in graph_list:
            cursor.execute(f"""
                SELECT date, close FROM {table_name}
                WHERE date >= %s ORDER BY date
            """, [graph])
            rows = cursor.fetchall()

            if not rows:
                print(f"No data available for {from_currency} to {to_currency} for the period starting {graph}")
                continue

            # Prepare data for plotting
            dates = [row[0] for row in rows]
            close_rates = [float(row[1]) for row in rows]

            # Plot the data
            plt.figure(figsize=(20, 5))
            plt.plot(dates, close_rates, label='Close Rate')
            plt.xlabel('Date')
            plt.ylabel('Exchange Rate')
            
            if graph == one_week_ago:
                name = "Last_week"
            elif graph == one_month_ago:
                name = "Last_month"
            elif graph == six_months_ago:
                name = "Last_6_months"
            elif graph == one_year_ago:
                name = "Last_year"
            elif graph == five_years_ago:
                name = "Last_5_years"
            
            plt.title(f'Exchange Rate: {from_currency} to {to_currency} {name}')
            plt.xticks(rotation=90)
            if name == "Last_5_years":
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
            elif name== "Last_year": 
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
            else: 
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.tight_layout()
            plt.legend()

            # Ensure the directory exists
            os.makedirs("myapp/Graphs", exist_ok=True)

            # Save the plot to a file
            file_path = f"myapp/Graphs/{from_currency}_to_{to_currency}_exchange_rate_{name}.png"
            try:
                plt.savefig(file_path)
                plt.close()
                print(f"Graph saved to {file_path}")
            except Exception as e:
                print(f"Failed to save graph: {e}")

class FetchHistoricalCurrencyExchangeRates(APIView):
   
    def get(self, request, from_currency, to_currency, *args, **kwargs):
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

        # Generate graphs after data insertion
        try:
            saveGraph(from_currency, to_currency, table_name)
        except Exception as e: 
            print(e)

        if will_insert:
            return Response({"Message": "Data fetched and stored successfully", "Result": data}, status=201)
        return Response({"Message": "Data fetched but no new data available",
                         "From": from_currency,
                         "To": to_currency,
                         "Data": data
                        }, status=201)

'''import os
import requests
from django.db import connection, transaction
from rest_framework.response import Response
from rest_framework.views import APIView
from dotenv import load_dotenv
import psycopg2.extras
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import io
import base64

# Load environment variables from the .env file
load_dotenv()

def saveGraph(from_currency, to_currency,table_name):
    with connection.cursor() as cursor:
        one_week_ago = datetime.now() - timedelta(days=8)
        one_month_ago = datetime.now() - timedelta(days=31)
        six_months_ago = datetime.now() - timedelta(days=181)
        one_year_ago = datetime.now() - timedelta(days=366)
        five_years_ago = datetime.now() - timedelta(days=(366*5))
        
        graph_list = [one_week_ago, one_month_ago, six_months_ago, one_year_ago, five_years_ago]
        
        for graph in graph_list:
            cursor.execute(f"""
                SELECT date, close FROM {table_name}
                WHERE date >= %s ORDER BY date
            """, [graph])
            rows = cursor.fetchall()

            # Prepare data for plotting
            dates = [row[0] for row in rows]
            close_rates = [float(row[1]) for row in rows]

            # Plot the data
            plt.figure(figsize=(20, 10))
            plt.plot(dates, close_rates, label='Close Rate')
            plt.xlabel('Date')
            plt.ylabel('Exchange Rate')
            
            if graph == one_week_ago:
                name= "Last_week"
            elif graph == one_month_ago:
                name= "Last_month"
            elif graph == six_months_ago:
                name= "Last_6_months"
            elif graph == one_year_ago:
                name= "Last_year"
            elif graph == five_years_ago:
                name= "Last_5_years"
            
            plt.title(f'Exchange Rate: {from_currency} to {to_currency} {name}')
            plt.xticks(rotation=90)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
            plt.tight_layout()
            plt.legend()

            # Save the plot to a file
            file_path = f"myapp/Graphs/{from_currency}_to_{to_currency}_exchange_rate_{name}.png"
            plt.savefig(file_path)
            plt.close()


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
        
        will_insert = True
        
        with connection.cursor() as cursor:
            if not (time_series.keys()):
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
                
            saveGraph(from_currency, to_currency, table_name)

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
                
                if not (time_series.keys()) or (max(time_series.keys())) <= latest_date.isoformat():
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
                                "Data": data
                            }, status=201)

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
