from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.Service.economicIndicatorService import AnnualIndicatorsService, MonthlyIndicatorService
from myapp.Serializers.economicIndicatorsSerializer import EconomicIndicatorsResponseSerializer
import logging
import requests

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
            # Store indicators and retrieve latest record in ADAGE format
            latest_data = AnnualIndicatorsService.store_annual_indicators()

            # Serialize response
            serializer = EconomicIndicatorsResponseSerializer(data=latest_data)
            if not serializer.is_valid():
                logger.error(f"Infalid ADAGE format: {serializer.errors}")
                return Response(
                    {"error": "Error formatting response data."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            return Response(serializer.data, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            return Response(
                {"error": "Failed to fetch data from external API."},
                status=status.HTTP_502_BAD_GATEWAY
                )

        except Exception as e:
            logger.exception(f"Error updating economic indicators: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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
