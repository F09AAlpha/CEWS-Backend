import os
import requests
import logging
from django.db import transaction
from myapp.models import AnnualEconomicIndicator
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
