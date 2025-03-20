import matplotlib.pyplot as plt
import io
from django.http import HttpResponse
from django.views import View
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from django.db import connection


class GraphView_lastmonth(View):
    def get(self, request, from_currency, to_currency):
        # Example: Fetch data for the last month
        one_month_ago = datetime.now() - timedelta(days=31)
        table_name = f"historical_exchange_rate_{from_currency.lower()}_{to_currency.lower()}"

        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT date, close FROM {table_name}
                WHERE date >= %s ORDER BY date
            """, [one_month_ago])
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
