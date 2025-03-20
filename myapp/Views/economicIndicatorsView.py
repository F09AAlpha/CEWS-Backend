from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.Service.economicIndicatorService import AnnualIndicatorsService, MonthlyIndicatorService
import logging

logger = logging.getLogger(__name__)


class StoreAnnualIndicatorsView(APIView):
    """
    API endpoint to fetch and store annual economic indicators (Real GDP & Inflation).
    """

    def post(self, request):
        """
        Fetches the latest economic indicators from Alpha Vantage and stires it in the database.

        Returns:
            Response: JSON response indicating success or failure.
        """
        try:
            AnnualIndicatorsService.store_annual_indicators()
            return Response({"message": "Annual economic indicators stored successfully."}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception(f"Error updating economic indicators: {str(e)}")
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StoreMonthlyIndicatorsView(APIView):
    """
    API endpoint to fetch and store monthly economic indicators (CPI, Unemployment Rate, Federal Funds Rate, Treasury Yield).
    """

    def post(self, request):
        """
        Triggers fetching and storing of monthly economic indicators.

        Returns:
            Response: JSON response indicating success or failure.
        """
        try:
            MonthlyIndicatorService.store_monthly_indicators()
            return Response(
                {"message": "Monthly economic indicators stored successfully."}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception(f"Error storing monthly economic indicators: {str(e)}")
            return Response(
                    {"error": "Failed to store monthly economic indicators."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
