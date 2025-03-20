import os
import requests
import logging
from django.db import transaction
from myapp.models import AnnualEconomicIndicator, MonthlyEconomicIndicator
from django.conf import settings

logger = logging.getLogger(__name__)


class AnnualIndicatorsService:
    # Service for fetching Annual Economic data from Aplha Vantage API
    API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY') or settings.ALPHA_VANTAGE_API_KEY
    BASE_URL = 'https://www.alphavantage.co/query'

    @staticmethod
    def fetch_annual_economic_data(indicator_type):
        """
          Fetches economic indicator data (Real GDP or Inflation) from Alpha Vantage.

          Args:
              indicator_type (str): Type of indicator ('REAL_GDP' or 'INFLATION')

          Returns:
              list: List of dictionaries containing 'date' and 'value'
        """
        params = {
            "function": indicator_type,
            "apikey": AnnualIndicatorsService.API_KEY
        }
        response = requests.get(AnnualIndicatorsService.BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json().get("data", [])
            return data
        else:
            logger.error(f"Failed to fetch {indicator_type} data: {response.text}")
            raise Exception(f"API request failed with status {response.status_code}")

    @staticmethod
    @transaction.atomic
    def store_annual_indicators():
        """
        Fetches and updates annual economic indicators (Real GDP & Inflation) in the database.
        Ensures only new data is added and maintains data integrity.
        """
        try:
            # Fetch latest data form Alpha Vantage
            gdp_data = AnnualIndicatorsService.fetch_annual_economic_data("REAL_GDP")
            inflation_data = AnnualIndicatorsService.fetch_annual_economic_data("INFLATION")

            # Convert data to a deictionary for easy lookup
            inflation_dict = {item["date"]: item["value"] for item in inflation_data}

            # Get the latest existing entyr form the database
            latest_entry = AnnualEconomicIndicator.objects.order_by("-date").first()
            latest_date = latest_entry.date if latest_entry else None

            new_entries = []
            for gdp_entry in gdp_data:
                date = gdp_entry["date"]
                real_gdp = float(gdp_entry["value"])
                inflation = float(inflation_dict.get(date, 0))

                # Only add new data
                if not latest_date or date > str(latest_date):
                    new_entries.append(AnnualEconomicIndicator(date=date, real_gdp=real_gdp, inflation=inflation))

            # Bulk insert new entries
            if new_entries:
                AnnualEconomicIndicator.objects.bulk_create(new_entries)
                logger.info(f"Inserted {len(new_entries)} new annual indicator records.")
            else:
                logger.info("No new annual indicators to store.")

        except Exception as e:
            logger.exception(f"Error storing annual indicators: {str(e)}")
            raise


class MonthlyIndicatorService:
    # Service for fetching Monthly Economic data from Aplha Vantage API
    API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY') or settings.ALPHA_VANTAGE_API_KEY
    BASE_URL = 'https://www.alphavantage.co/query'

    @staticmethod
    def fetch_monthly_economic_data(indicator_type):
        """
          Fetches economic indicator data Unemployment, CPI, Federal Funds, etc. from Alpha Vantage.

          Args:
              indicator_type (str): Type of indicator ('UNEMPLOYMENT' or 'CPI')

          Returns:
              list: List of dictionaries containing 'date' and 'value'
        """
        params = {
            "function": indicator_type,
            "apikey": MonthlyIndicatorService.API_KEY
        }
        response = requests.get(MonthlyIndicatorService.BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json().get("data", [])
            return data
        else:
            logger.error(f"Failed to fetch {indicator_type} data: {response.text}")
            raise Exception(f"API request failed with status {response.status_code}")

    @staticmethod
    def store_monthly_indicators():
        """
        Fetches and stores monthly economic indicators (CPI, Unemployment Rate, Federal Funds Rate, Treasury Yield).
        Only stores new data if it is more recent than the latest existing entry.
        """
        try:
            # Fetch the latest stored date
            latest_entry = MonthlyEconomicIndicator.objects.order_by('-date').first()
            latest_stored_date = latest_entry.date if latest_entry else None

            # Fetch data from Alpha Vantage
            cpi_data = MonthlyIndicatorService.fetch_monthly_economic_data("CPI")
            unemployment_data = MonthlyIndicatorService.fetch_monthly_economic_data("UNEMPLOYMENT")
            federal_funds_data = MonthlyIndicatorService.fetch_monthly_economic_data("FEDERAL_FUNDS_RATE")
            treasury_yield_data = MonthlyIndicatorService.fetch_monthly_economic_data("TREASURY_YIELD")

            # Convert API data into a dictionary keyed by date
            indicator_data = {}
            for dataset, key in [
                (cpi_data, "cpi"),
                (unemployment_data, "unemployment_rate"),
                (federal_funds_data, "federal_funds_rate"),
                (treasury_yield_data, "treasury_yield")
            ]:
                for entry in dataset:
                    date = entry["date"]
                    value = float(entry["value"])
                    if date not in indicator_data:
                        indicator_data[date] = {}
                    indicator_data[date][key] = value

            # Insert only new data
            new_entries = []
            for date, values in indicator_data.items():
                if latest_stored_date and date <= latest_stored_date.strftime('%Y-%m-%d'):
                    continue  # Skip older or already stored data

                new_entries.append(
                    MonthlyEconomicIndicator(
                        date=date,
                        cpi=values.get("cpi"),
                        unemployment_rate=values.get("unemployment_rate"),
                        federal_funds_rate=values.get("federal_funds_rate"),
                        treasury_yield=values.get("treasury_yield")
                    )
                )

            if new_entries:
                MonthlyEconomicIndicator.objects.bulk_create(new_entries)
                logger.info(f"Stored {len(new_entries)} new monthly economic indicator records.")
            else:
                logger.info("No new monthly economic indicator data to store.")

        except Exception as e:
            logger.exception(f"Error storing monthly economic indicators: {str(e)}")
            raise

    @staticmethod
    def get_value_by_date(data, date_str):
        """Helper function to retrieve a value from an API dataset by date."""
        for entry in data:
            if entry["date"] == date_str:
                return entry["value"]
        return None
