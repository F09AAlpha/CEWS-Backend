from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from myapp.Service.economicIndicatorService import AnnualIndicatorsService
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
