# analytics/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
from myapp.Service.volatilityService import VolatilityService
from myapp.Serializers.volatilitySerializer import VolatilityAnalysisSerializer
import logging

logger = logging.getLogger(__name__)


class VolatilityAnalysisView(APIView):
    """
    API endpoint for currency volatility analysis
    """

    # Cache results for 1 hour (3600 seconds)
    @method_decorator(cache_page(3600))
    def get(self, request, base, target):
        try:
            # Get days parameter (default: 30)
            days = int(request.query_params.get('days', 30))

            # Limit analysis period to reasonable range
            if days < 7:
                days = 7
            elif days > 365:
                days = 365

            # Perform volatility analysis
            volatility_service = VolatilityService()
            analysis_result = volatility_service.calculate_volatility(base.upper(), target.upper(), days)

            # Validate with serializer
            serializer = VolatilityAnalysisSerializer(data=analysis_result)
            if serializer.is_valid():
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except ValueError as e:
            logger.error(f"Value error in volatility analysis: {str(e)}")
            return Response(
                {"detail": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            return Response(
                {"detail": "An error occurred during volatility analysis."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
