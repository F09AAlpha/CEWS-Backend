import io
import logging as logger
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from django.db import connection
from django.http import HttpResponse
from django.views import View
matplotlib.use('Agg')  # Set the backend to Agg which is non-interactive


class GraphView_lastweek(View):
    def get(self, request, from_currency, to_currency):
        try:
            # Example: Fetch data for the last month
            one_week_ago = datetime.now() - timedelta(days=8)
            table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

            logger.info(f"Fetching data for {from_currency} to {to_currency} from table {table_name}")

            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT date, close FROM {table_name}
                    WHERE date >= %s ORDER BY date
                """, [one_week_ago])
                rows = cursor.fetchall()

            if not rows:
                logger.warning(f"No data available for {from_currency} to {to_currency} for the last week")
                return HttpResponse("No data available for the last week.", status=404)

            # Prepare data for plotting
            dates = [row[0] for row in rows]
            close_rates = [float(row[1]) for row in rows]

            # Plot the data
            plt.figure(figsize=(10, 5))
            plt.plot(dates, close_rates, label='Close Rate')
            plt.xlabel('Date')
            plt.ylabel('Exchange Rate')
            plt.title(f'Exchange Rate: {from_currency} to {to_currency} Last week')
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.tight_layout()
            plt.legend()

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            # Return the image as an HTTP response
            return HttpResponse(buf, content_type='image/png')
        except Exception as e:
            logger.error(f"Error generating last week graph for {from_currency} to {to_currency}: {str(e)}")
            return HttpResponse(f"Error generating graph: {str(e)}", status=500)


class GraphView_lastmonth(View):
    def get(self, request, from_currency, to_currency):
        try:
            # Example: Fetch data for the last month
            one_month_ago = datetime.now() - timedelta(days=31)
            table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

            logger.info(f"Fetching data for {from_currency} to {to_currency} from table {table_name}")

            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT date, close FROM {table_name}
                    WHERE date >= %s ORDER BY date
                """, [one_month_ago])
                rows = cursor.fetchall()

            if not rows:
                logger.warning(f"No data available for {from_currency} to {to_currency} for the last month")
                return HttpResponse("No data available for the last month.", status=404)

            # Prepare data for plotting
            dates = [row[0] for row in rows]
            close_rates = [float(row[1]) for row in rows]

            # Plot the data
            plt.figure(figsize=(10, 5))
            plt.plot(dates, close_rates, label='Close Rate')
            plt.xlabel('Date')
            plt.ylabel('Exchange Rate')
            plt.title(f'Exchange Rate: {from_currency} to {to_currency} Last Month')
            plt.xticks(rotation=90)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.tight_layout()
            plt.legend()

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            # Return the image as an HTTP response
            return HttpResponse(buf, content_type='image/png')
        except Exception as e:
            logger.error(f"Error generating graph for {from_currency} to {to_currency}: {str(e)}")
            return HttpResponse(f"Error generating graph: {str(e)}", status=500)


class GraphView_last6months(View):
    def get(self, request, from_currency, to_currency):
        # Example: Fetch data for the last month
        six_months_ago = datetime.now() - timedelta(days=180)
        table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT date, close FROM {table_name}
                WHERE date >= %s ORDER BY date
            """, [six_months_ago])
            rows = cursor.fetchall()

        if not rows:
            return HttpResponse("No data available for the last month.", status=404)

        # Prepare data for plotting
        dates = [row[0] for row in rows]
        close_rates = [float(row[1]) for row in rows]

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(dates, close_rates, label='Close Rate')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.title(f'Exchange Rate: {from_currency} to {to_currency} Last 6 months')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
        plt.tight_layout()
        plt.legend()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Return the image as an HTTP response
        return HttpResponse(buf, content_type='image/png')


class GraphView_lastyear(View):
    def get(self, request, from_currency, to_currency):
        # Example: Fetch data for the last month
        one_year_ago = datetime.now() - timedelta(days=365)
        table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT date, close FROM {table_name}
                WHERE date >= %s ORDER BY date
            """, [one_year_ago])
            rows = cursor.fetchall()

        if not rows:
            return HttpResponse("No data available for the last year.", status=404)

        # Prepare data for plotting
        dates = [row[0] for row in rows]
        close_rates = [float(row[1]) for row in rows]

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(dates, close_rates, label='Close Rate')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.title(f'Exchange Rate: {from_currency} to {to_currency} Last year')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
        plt.tight_layout()
        plt.legend()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Return the image as an HTTP response
        return HttpResponse(buf, content_type='image/png')


class GraphView_last5years(View):
    def get(self, request, from_currency, to_currency):
        try:
            # Example: Fetch data for the last month
            five_years_ago = datetime.now() - timedelta(days=(365*5))
            table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

            logger.info(f"Fetching data for {from_currency} to {to_currency} from table {table_name}")

            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT date, close FROM {table_name}
                    WHERE date >= %s ORDER BY date
                """, [five_years_ago])
                rows = cursor.fetchall()

            if not rows:
                logger.warning(f"No data available for {from_currency} to {to_currency} for the last 5 years")
                return HttpResponse("No data available for the last 5 years.", status=404)

            # Prepare data for plotting
            dates = [row[0] for row in rows]
            close_rates = [float(row[1]) for row in rows]

            # Plot the data
            plt.figure(figsize=(10, 5))
            plt.plot(dates, close_rates, label='Close Rate')
            plt.xlabel('Date')
            plt.ylabel('Exchange Rate')
            plt.title(f'Exchange Rate: {from_currency} to {to_currency} Last 5 years')
            plt.xticks(rotation=90)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=183))
            plt.tight_layout()
            plt.legend()

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            # Return the image as an HTTP response
            return HttpResponse(buf, content_type='image/png')
        except Exception as e:
            logger.error(f"Error generating 5-year graph for {from_currency} to {to_currency}: {str(e)}")
            return HttpResponse(f"Error generating graph: {str(e)}", status=500)
