from django.test import TestCase
from unittest.mock import MagicMock, patch
from myapp.Models.economicIndicatorsModel import AnnualEconomicIndicator, MonthlyEconomicIndicator
from myapp.Service.economicIndicatorService import AnnualIndicatorsService, MonthlyIndicatorService

# import pytz


class AnnualIndicatorUnitTest(TestCase):
    """ Tests for the Annual Economic Indicators Service """

    """ Tests for fetch_annual_economic_data function """
    @patch('myapp.Service.economicIndicatorService.requests.get')
    def test_fetch_annual_economic_data_success(self, mock_get):
        """ Test successful retrival of annual economic data """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"date": "2024-01-01", "value": "23303.5"}]}
        mock_get.return_value = mock_response

        result = AnnualIndicatorsService.fetch_annual_economic_data("REAL_GDP")
        self.assertEqual(result, [{"date": "2024-01-01", "value": "23303.5"}])

    @patch('myapp.Service.economicIndicatorService.requests.get')
    def test_fetch_annual_economic_data_failure(self, mock_get):
        """ Test failed retrival of annual economic data """
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            AnnualIndicatorsService.fetch_annual_economic_data("REAL_GDP")
        self.assertIn("API request failed", str(context.exception))

    @patch('myapp.Service.economicIndicatorService.requests.get')
    def test_fetch_annual_economic_data_invalid_format(self, mock_get):
        """ Test unexpected data recieved during retrival from external API"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected_key": []}
        mock_get.return_value = mock_response

        result = AnnualIndicatorsService.fetch_annual_economic_data("REAL_GDP")
        self.assertEqual(result, [])  # Expecting empty list since "data" key is missing

    """ Tests for store_annual_indicators function """
    @patch('myapp.Service.economicIndicatorService.AnnualIndicatorsService.fetch_annual_economic_data')
    def test_store_annual_indicators_success(self, mock_fetch):
        """Test that store_annual_indicators correctly stores data and returns proper ADAGE 3.0 format"""

        # Setup mock return values
        mock_fetch.side_effect = [
            # GDP data
            [
                {"date": "2023-01-01", "value": "24989.50"},
                {"date": "2022-01-01", "value": "23315.08"}
            ],
            # Inflation data
            [
                {"date": "2023-01-01", "value": "3.4"},
                {"date": "2022-01-01", "value": "8.0"}
            ]
        ]

        # Call the method being tested
        result = AnnualIndicatorsService.store_annual_indicators()

        # Verify both API endpoints were called with correct parameters
        mock_fetch.assert_any_call("REAL_GDP")
        mock_fetch.assert_any_call("INFLATION")

        # Check database records were created
        self.assertEqual(AnnualEconomicIndicator.objects.count(), 2)

        # Verify the latest record has correct values
        latest = AnnualEconomicIndicator.objects.order_by("-date").first()
        self.assertEqual(str(latest.date), "2023-01-01")
        self.assertEqual(float(latest.real_gdp), 24989.50)
        self.assertEqual(float(latest.inflation), 3.4)

        # Verify ADAGE 3.0 format
        self.assertEqual(result["data_source"], "Alpha Vantage")
        self.assertEqual(result["dataset_type"], "annual_economic_indicators")
        self.assertEqual(result["dataset_id"], "annual-indicators-2023-01-01")

        # Check time object
        self.assertIn("time_object", result)
        self.assertIn("timestamp", result["time_object"])
        self.assertEqual(result["time_object"]["timezone"], "UTC")

        # Check events array
        self.assertIn("events", result)
        self.assertEqual(len(result["events"]), 1)

        event = result["events"][0]
        self.assertEqual(event["event_type"], "economic_indicator")
        self.assertIn("event_id", event)

        # Check event attributes
        self.assertEqual(float(event["attributes"]["real_gdp"]), 24989.50)
        self.assertEqual(float(event["attributes"]["inflation"]), 3.4)
        self.assertEqual(event["attributes"]["source"], "Alpha Vantage")

        # Check event time object
        self.assertEqual(event["time_object"]["timestamp"], "2023-01-01")
        self.assertEqual(event["time_object"]["duration"], 365)
        self.assertEqual(event["time_object"]["duration_unit"], "days")
        self.assertEqual(event["time_object"]["timezone"], "UTC")

    @patch('myapp.Service.economicIndicatorService.AnnualIndicatorsService.fetch_annual_economic_data')
    def test_store_annual_indicators_with_existing_data(self, mock_fetch):
        """Test that store_annual_indicators only adds new data when database has existing entries"""

        # Create existing record
        AnnualEconomicIndicator.objects.create(
            date="2022-01-01",
            real_gdp=23315.08,
            inflation=8.0
        )

        # Setup mock return values
        mock_fetch.side_effect = [
            # GDP data
            [
                {"date": "2023-01-01", "value": "24989.50"},
                {"date": "2022-01-01", "value": "23315.08"}  # This should be skipped
            ],
            # Inflation data
            [
                {"date": "2023-01-01", "value": "3.4"},
                {"date": "2022-01-01", "value": "8.0"}  # This should be skipped
            ]
        ]

        # Call the method being tested
        result = AnnualIndicatorsService.store_annual_indicators()

        # Should only have 2 records total (1 new + 1 existing)
        self.assertEqual(AnnualEconomicIndicator.objects.count(), 2)

        # Latest should be the new record
        latest = AnnualEconomicIndicator.objects.order_by("-date").first()
        self.assertEqual(str(latest.date), "2023-01-01")

        # Verify ADAGE data is for the latest record
        self.assertEqual(result["dataset_id"], "annual-indicators-2023-01-01")
        self.assertEqual(float(result["events"][0]["attributes"]["real_gdp"]), 24989.50)

    @patch('myapp.Service.economicIndicatorService.AnnualIndicatorsService.fetch_annual_economic_data')
    def test_store_annual_indicators_handles_api_error(self, mock_fetch):
        """Test that store_annual_indicators properly handles API errors"""

        # Make the mock raise an exception
        mock_fetch.side_effect = Exception("API Connection Error")

        # Should raise the exception
        with self.assertRaises(Exception):
            AnnualIndicatorsService.store_annual_indicators()

        # Verify no records were created
        self.assertEqual(AnnualEconomicIndicator.objects.count(), 0)


class MonthlyIndicatorUnitTest(TestCase):
    """ Tests for the Monthly Economic Indicators Service """

    """ Tests for fetch_monthly_economic_data function """
    @patch('myapp.Service.economicIndicatorService.requests.get')
    def test_fetch_monthly_economic_data_success(self, mock_get):
        """ Test successful retrieval of monthly economic data """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"date": "2024-03-01", "value": "4.1"}]}
        mock_get.return_value = mock_response

        result = MonthlyIndicatorService.fetch_monthly_economic_data("UNEMPLOYMENT")
        self.assertEqual(result, [{"date": "2024-03-01", "value": "4.1"}])

    @patch('myapp.Service.economicIndicatorService.requests.get')
    def test_fetch_monthly_economic_data_failure(self, mock_get):
        """ Test failed retrieval of monthly economic data """
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            MonthlyIndicatorService.fetch_monthly_economic_data("CPI")
        self.assertIn("API request failed", str(context.exception))

    @patch('myapp.Service.economicIndicatorService.requests.get')
    def test_fetch_monthly_economic_data_invalid_format(self, mock_get):
        """ Test unexpected data received during retrieval from external API"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected_key": []}
        mock_get.return_value = mock_response

        result = MonthlyIndicatorService.fetch_monthly_economic_data("UNEMPLOYMENT")
        self.assertEqual(result, [])  # Expecting empty list since "data" key is missing

    """ Tests for store_monthly_indicators function """
    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.fetch_monthly_economic_data')
    def test_store_monthly_indicators_success(self, mock_fetch):
        """Test that store_monthly_indicators correctly stores data and returns proper ADAGE 3.0 format"""

        # Setup mock return values for each API call
        mock_fetch.side_effect = [
            # CPI data
            [
                {"date": "2024-03-01", "value": "307.8"},
                {"date": "2024-02-01", "value": "305.6"}
            ],
            # Unemployment data
            [
                {"date": "2024-03-01", "value": "4.1"},
                {"date": "2024-02-01", "value": "4.2"}
            ],
            # Federal Funds Rate data
            [
                {"date": "2024-03-01", "value": "5.33"},
                {"date": "2024-02-01", "value": "5.33"}
            ],
            # Treasury Yield data
            [
                {"date": "2024-03-01", "value": "4.35"},
                {"date": "2024-02-01", "value": "4.30"}
            ]
        ]

        # Call the method being tested
        result = MonthlyIndicatorService.store_monthly_indicators()

        # Verify all API endpoints were called with correct parameters
        mock_fetch.assert_any_call("CPI")
        mock_fetch.assert_any_call("UNEMPLOYMENT")
        mock_fetch.assert_any_call("FEDERAL_FUNDS_RATE")
        mock_fetch.assert_any_call("TREASURY_YIELD")

        # Check database records were created (should have 2 entries)
        self.assertEqual(MonthlyEconomicIndicator.objects.count(), 2)

        # Verify the latest record has correct values
        latest = MonthlyEconomicIndicator.objects.order_by("-date").first()
        self.assertEqual(str(latest.date), "2024-03-01")
        self.assertEqual(float(latest.cpi), 307.8)
        self.assertEqual(float(latest.unemployment_rate), 4.1)
        self.assertEqual(float(latest.federal_funds_rate), 5.33)
        self.assertEqual(float(latest.treasury_yield), 4.35)

        # Verify ADAGE 3.0 format
        self.assertEqual(result["data_source"], "Alpha Vantage")
        self.assertEqual(result["dataset_type"], "monthly_economic_indicators")
        self.assertEqual(result["dataset_id"], "monthly-indicators-2024-03-01")

        # Check time object
        self.assertIn("time_object", result)
        self.assertIn("timestamp", result["time_object"])
        self.assertEqual(result["time_object"]["timezone"], "UTC")

        # Check events array
        self.assertIn("events", result)
        self.assertEqual(len(result["events"]), 1)

        event = result["events"][0]
        self.assertEqual(event["event_type"], "economic_indicator")
        self.assertIn("event_id", event)

        # Check event attributes
        self.assertEqual(float(event["attributes"]["cpi"]), 307.8)
        self.assertEqual(float(event["attributes"]["unemployment_rate"]), 4.1)
        self.assertEqual(float(event["attributes"]["federal_funds_rate"]), 5.33)
        self.assertEqual(float(event["attributes"]["treasury_yield"]), 4.35)
        self.assertEqual(event["attributes"]["source"], "Alpha Vantage")

        # Check event time object
        self.assertEqual(event["time_object"]["timestamp"], "2024-03-01")
        self.assertEqual(event["time_object"]["duration"], 1)
        self.assertEqual(event["time_object"]["duration_unit"], "month")
        self.assertEqual(event["time_object"]["timezone"], "UTC")

    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.fetch_monthly_economic_data')
    def test_store_monthly_indicators_with_existing_data(self, mock_fetch):
        """Test that store_monthly_indicators only adds new data when database has existing entries"""

        # Create existing record
        MonthlyEconomicIndicator.objects.create(
            date="2024-02-01",
            cpi=305.6,
            unemployment_rate=4.2,
            federal_funds_rate=5.33,
            treasury_yield=4.30
        )

        # Setup mock return values
        mock_fetch.side_effect = [
            # CPI data
            [
                {"date": "2024-03-01", "value": "307.8"},
                {"date": "2024-02-01", "value": "305.6"}  # This should be skipped
            ],
            # Unemployment data
            [
                {"date": "2024-03-01", "value": "4.1"},
                {"date": "2024-02-01", "value": "4.2"}  # This should be skipped
            ],
            # Federal Funds Rate data
            [
                {"date": "2024-03-01", "value": "5.33"},
                {"date": "2024-02-01", "value": "5.33"}  # This should be skipped
            ],
            # Treasury Yield data
            [
                {"date": "2024-03-01", "value": "4.35"},
                {"date": "2024-02-01", "value": "4.30"}  # This should be skipped
            ]
        ]

        # Call the method being tested
        result = MonthlyIndicatorService.store_monthly_indicators()

        # Should only have 2 records total (1 new + 1 existing)
        self.assertEqual(MonthlyEconomicIndicator.objects.count(), 2)

        # Latest should be the new record
        latest = MonthlyEconomicIndicator.objects.order_by("-date").first()
        self.assertEqual(str(latest.date), "2024-03-01")

        # Verify ADAGE data is for the latest record
        self.assertEqual(result["dataset_id"], "monthly-indicators-2024-03-01")
        self.assertEqual(float(result["events"][0]["attributes"]["cpi"]), 307.8)
        self.assertEqual(float(result["events"][0]["attributes"]["unemployment_rate"]), 4.1)

    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.fetch_monthly_economic_data')
    def test_store_monthly_indicators_no_new_data(self, mock_fetch):
        """Test that store_monthly_indicators handles case when no new data is available"""

        # Create existing record with most recent date
        MonthlyEconomicIndicator.objects.create(
            date="2024-03-01",
            cpi=307.8,
            unemployment_rate=4.1,
            federal_funds_rate=5.33,
            treasury_yield=4.35
        )

        # Setup mock return values with same or older dates
        mock_fetch.side_effect = [
            # CPI data - same date as existing
            [
                {"date": "2024-03-01", "value": "307.8"},
                {"date": "2024-02-01", "value": "305.6"}
            ],
            # Unemployment data
            [
                {"date": "2024-03-01", "value": "4.1"},
                {"date": "2024-02-01", "value": "4.2"}
            ],
            # Federal Funds Rate data
            [
                {"date": "2024-03-01", "value": "5.33"},
                {"date": "2024-02-01", "value": "5.33"}
            ],
            # Treasury Yield data
            [
                {"date": "2024-03-01", "value": "4.35"},
                {"date": "2024-02-01", "value": "4.30"}
            ]
        ]

        # Call the method being tested
        result = MonthlyIndicatorService.store_monthly_indicators()

        # Should still only have 1 record (no new ones added)
        self.assertEqual(MonthlyEconomicIndicator.objects.count(), 1)

        # Verify ADAGE data is still for the latest record
        self.assertEqual(result["dataset_id"], "monthly-indicators-2024-03-01")

    @patch('myapp.Service.economicIndicatorService.MonthlyIndicatorService.fetch_monthly_economic_data')
    def test_store_monthly_indicators_handles_api_error(self, mock_fetch):
        """Test that store_monthly_indicators properly handles API errors"""

        # Make the mock raise an exception
        mock_fetch.side_effect = Exception("API Connection Error")

        # Should raise the exception
        with self.assertRaises(Exception):
            MonthlyIndicatorService.store_monthly_indicators()

        # Verify no records were created
        self.assertEqual(MonthlyEconomicIndicator.objects.count(), 0)

    """ Tests for get_value_by_date helper function """
    def test_get_value_by_date_found(self):
        """Test that get_value_by_date returns correct value when date is found"""
        test_data = [
            {"date": "2024-03-01", "value": "307.8"},
            {"date": "2024-02-01", "value": "305.6"}
        ]

        result = MonthlyIndicatorService.get_value_by_date(test_data, "2024-03-01")
        self.assertEqual(result, "307.8")

    def test_get_value_by_date_not_found(self):
        """Test that get_value_by_date returns None when date is not found"""
        test_data = [
            {"date": "2024-03-01", "value": "307.8"},
            {"date": "2024-02-01", "value": "305.6"}
        ]

        result = MonthlyIndicatorService.get_value_by_date(test_data, "2024-01-01")
        self.assertIsNone(result)
